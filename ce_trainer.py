from __future__ import absolute_import
import logging
import os, sys
import rain
from rain.model import Trainer
from rain.bucket_io import build_data, DataPartioner
from rain.optimizer import NewAdam
from rain.evaluate import WarmupScheduler, KingScheduler, Speedometer, BatchEndParam
from rain.metric import GenTranMetric
from rain.utils import collate_torch_dict
from rain.initializer import initialize_model, save_pytorch_model, initialize_model_from_pytorch_v2
from rain import xconfig
import time
import copy
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import re
import pdb
from loader_profiler import LoaderProfiler


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

rank = int(os.environ['RANK'])
torch.set_printoptions(precision=7, threshold=1000, edgeitems=3, linewidth=80)
parser = argparse.ArgumentParser("OCR Training")
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

# os.environ['CUDA_LAUNCH_BLOCKING'] = 1 
# set logger
# 
if rank == 0:
    if not os.path.exists('./train_log'):
        os.mkdir('./train_log')
else:
    time.sleep(1)

logging.basicConfig(
    level=logging.INFO,
    format='Node[{}] %(asctime)s[%(levelname)s] %(name)s -%(message)s'.format(rank),
    handlers=[logging.FileHandler(os.path.join('train_log/train-debug-rank[{}]-.log'.format(int(os.environ["RANK"])))),
              logging.StreamHandler()]
)
logger = logging.getLogger()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def average_update(model):
    size = float(dist.get_world_size())
    for k, v in model.named_parameters():
        dist.all_reduce(v.data, op=dist.ReduceOp.SUM)
        v.data /= size
    for k, v in model.named_buffers():
        if 'num_batches_tracked' in k:
            continue
        dist.all_reduce(v.data, op=dist.ReduceOp.SUM)
        v.data /= size


def bmuf_update(model, global_models, momentums, rescale_grad=1.0):
    average_update(model)
    for (k, v), global_param, momentum in zip(model.named_parameters(), global_models, momentums):
        grad = v.data * rescale_grad - global_param
        momentum *= xconfig.bmuf_params["bm"]
        global_param -= momentum
        momentum += xconfig.bmuf_params["blr"] * grad
        global_param += (1.0 + xconfig.bmuf_params["bm"]) * momentum
        v.detach().copy_(global_param.detach())


def validate(model, data_loader, epoch_batch=1, scheduler_metric='ce'):
    logger.info("begin eval")
    model.eval()
    metrics = GenTranMetric(name="valid")
    with torch.no_grad():
        for ibatch, data_dict in enumerate(data_loader):
            if ibatch % 500 == 0:
                logger.info(str(ibatch))
            for key in data_dict:
                value = data_dict[key]
                if not isinstance(value, torch.Tensor):
                    continue
                if key.find("mask") != -1:
                    data_dict[key] = value.cuda().detach()
                else:
                    data_dict[key] = value.cuda()
            target = data_dict["target"]  #[B, l_tgt]
            target_mask = data_dict["target_mask"]  #[B, l_tgt]
            target_branch = data_dict["branch_target"]  #[B, l_branch, l_bond]

            if len(target) == 0 or len(target_mask) == 0 or len(target_branch) == 0:
                continue # skip blank data
            
            try:
                preds = model(data_dict)
            except RuntimeError as e:
                print("[valid] abnormal batch, skip")
                continue
            except BaseException as e:
                print("[valid] abnormal batch, skip")
                continue
            for metric in metrics:
                metric.update([target, target_branch, target_mask], preds)
            if (ibatch + 1) == epoch_batch:
                break
    name_values = []
    valid_msg = 'Epoch[%d] '
    msgs = []
    scheduler_metric_value = None
    for metric in metrics:
        name_values += metric.get()
        msgs.append("%s=%f")
        if scheduler_metric in name_values[0]:
            scheduler_metric_value = name_values[1]
    valid_msg += ("\t".join(msgs))
    model.train()
    logger.info("end eval")
    return name_values, valid_msg, scheduler_metric_value


def check_load_epoch(model_prefix):
    logger.info("check load epoch by prefix=%s" % (model_prefix))
    baseDir = os.path.dirname(model_prefix)
    if not os.path.exists(baseDir):
        return None
    baseName = os.path.basename(model_prefix)
    filenames = os.listdir(baseDir)
    load_epoch = -1
    for i, filename in enumerate(filenames):
        if filename.find(baseName) != -1:
            spts = re.split("[.-]", filename)
            try:
                cur_epoch = int(spts[-2])
                if cur_epoch > load_epoch:
                    load_epoch = cur_epoch
            except:
                pass
    if load_epoch == -1:
        load_epoch = None
    return load_epoch


def train(global_rank=0, local_rank=0, world_size=1):
    torch.cuda.set_device(local_rank)
    setup_seed(xconfig.seed)
    trainer_model = Trainer()
    
    initialize_model(trainer_model, xconfig.model_prefix, xconfig.model_type, xconfig.load_epoch, global_rank, xconfig.allow_missing)
    if xconfig.load_param_path is not None:
        initialize_model_from_pytorch_v2(trainer_model, xconfig.load_param_path, allow_missing=True)
    trainer_model.cuda()
    
    trainer_opt = NewAdam(trainer_model.parameters(), lr=xconfig.learning_rate, eps=1e-8, weight_decay=xconfig.weight_decay)
    val_scheduler_dict = copy.deepcopy(xconfig.val_scheduler_dict)
    warmup_scheduler = WarmupScheduler(
        trainer_opt,
        start_lr=val_scheduler_dict["warmup_start_lr"],
        stop_lr=xconfig.learning_rate,
        step=val_scheduler_dict["warmup_step"],
        frequent=val_scheduler_dict["warmup_disp_freq"]
    )
    king_scheduler = KingScheduler(trainer_opt, val_scheduler_dict)
    
    # ==============Training Loop=================== #
    begin_epoch = xconfig.load_epoch + 1 if xconfig.load_epoch else 0
    num_epoch = xconfig.num_epochs
    batch_size = xconfig.fix_batch_size if xconfig.fix_batch_size is not None else 10
    for i in range(begin_epoch):
        king_scheduler.__call__(None)

    # init only once
    train_set = build_data(
        lrcfile=xconfig.train_lrc,
        keyfile=xconfig.train_key,
        ignorefile=xconfig.train_ignore,
        max_h=xconfig.max_height,
        max_w=xconfig.max_width,
        max_l=xconfig.max_length,
        fix_batch_size=xconfig.fix_batch_size,
        max_batch_size=xconfig.max_batch_size,
        max_image_size=xconfig.max_image_size,
        seed=xconfig.seed,
        do_shuffle=True,
        use_all=True,
        last_method='fill',
        one_key=False,
        return_name=True,
        do_test=False,
        image_list_file=None
    )
    print('Done train set')
    
    # auto set epoch batch
    if global_rank == 0:
        batch_num = len(train_set)
        epoch_batch = int(float(batch_num) / xconfig.data_divide_num)
        if xconfig.use_bmuf:
            epoch_batch = int(float(batch_num) / (xconfig.num_gpus * xconfig.data_divide_num))  # 16000 batch as one epoch
        
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
        if world_size > 1:
            dist.broadcast_object_list([epoch_batch], 0)
    else:
        assert dist.is_initialized(), "dist must be init when rank > 0"
        obj_list = [None]
        dist.broadcast_object_list(obj_list, 0)
        epoch_batch = obj_list[0]
    epoch_batch = int(epoch_batch / 2) # epoch_batch减半
    xconfig.epoch_batch = epoch_batch
    print("Set xconfig.epoch_batch=%d" % epoch_batch)
    


    valid_set = build_data(
        lrcfile=xconfig.dev_lrc,
        keyfile=xconfig.dev_key,
        max_h=xconfig.test_max_height,
        max_w=xconfig.test_max_width,
        max_l=xconfig.test_max_length,
        fix_batch_size=xconfig.test_fix_batch_size,
        max_batch_size=xconfig.max_batch_size,
        max_image_size=xconfig.max_image_size,
        seed=xconfig.seed,
        do_shuffle=True,
        use_all=False,
        last_method='fill',
        one_key=False,
        return_name=False,
        image_list_file=None,
        do_test=True
    )
    valid_partition = DataPartioner(valid_set, size=1, rank=0, max_batch_one_epoch=xconfig.val_epoch_batch)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_partition, batch_size=1, num_workers=4, collate_fn=collate_torch_dict, shuffle=False)
    
    # bmuf global states
    momentums = []
    global_models = []
    for param in trainer_model.parameters():
        temp = torch.zeros_like(param, requires_grad=False)
        temp.copy_(param.data)
        global_models.append(temp)
        momentums.append(torch.zeros_like(param, requires_grad=False))

    # start training
    epoch = begin_epoch
    tic = time.time()
    ibatch = 0
    
    if rank == 0:
        torch.save(trainer_model.state_dict(), 'ed_mem_tester.pt')
        model_size = os.path.getsize('ed_mem_tester.pt') / (2**20)
        logger.info('model size = %d M' % (model_size))
        os.system('rm %s' % ('ed_mem_tester.pt'))

    scaler = GradScaler() if xconfig.train_amp else None
    
    cur_key_ind = 0
    while epoch < num_epoch:
        cur_train_lrc = xconfig.train_lrc % (xconfig.train_key_inds[cur_key_ind % len(xconfig.train_key_inds)]) if '%d' in xconfig.train_lrc else xconfig.train_lrc
        cur_train_key = xconfig.train_key % (xconfig.train_key_inds[cur_key_ind % len(xconfig.train_key_inds)]) if '%d' in xconfig.train_key else xconfig.train_key
        cur_key_ind += 1
        train_set = build_data(
            lrcfile=cur_train_lrc,
            keyfile=cur_train_key,
            max_h=xconfig.max_height,
            max_w=xconfig.max_width,
            max_l=xconfig.max_length,
            fix_batch_size=xconfig.fix_batch_size,
            max_batch_size=xconfig.max_batch_size,
            max_image_size=xconfig.max_image_size,
            seed=xconfig.seed + cur_key_ind,
            do_shuffle=True,
            use_all=True,
            last_method='fill',
            one_key=False,
            return_name=True,
            image_list_file=None
        )
        logger.info('Loading train set: %s' % cur_train_key)
        train_partition = DataPartioner(train_set, size=world_size, rank=global_rank, max_batch_one_epoch=xconfig.epoch_batch, seed=cur_key_ind)
        train_loader = torch.utils.data.DataLoader(dataset=train_partition, batch_size=1, num_workers=6, collate_fn=collate_torch_dict, shuffle=False, prefetch_factor=8)
        
        speed_metric = Speedometer(batch_size, xconfig.epoch_batch, xconfig.disp_batches, opt=trainer_opt)
        #for data,data_mask,target_ori,target_mask_ori,train_names_list in train_loader:
        for _, data_dict in LoaderProfiler(train_loader, 500, True, logger=logger):
            for key in data_dict:
                value = data_dict[key]
                if not isinstance(value, torch.Tensor):
                    continue
                if key.find("mask") != -1:
                    data_dict[key] = value.cuda().detach()
                else:
                    data_dict[key] = value.cuda()
            train_names_list = data_dict["names_list"]

            if len(train_names_list) == 0: # skip blank data
                continue
            
            trainer_opt.zero_grad()
            
            try:
                if xconfig.train_amp:
                    #check_grad = trainer_model.forward_backward_amp(data,data_mask,target_ori,target_mask_ori,scaler,trainer_opt)
                    check_grad = False
                    if check_grad:
                        scaler.step(trainer_opt)
                        scaler.update()
                    else:
                        trainer_opt.zero_grad()
                        scaler.step(trainer_opt)
                        scaler.update()
                else:
                    # _string_cs = xconfig.vocab.indices2words(data_dict['target'][0].tolist())
                    check_grad = trainer_model.forward_backward(data_dict) # forward
                    if check_grad:
                        trainer_opt.step()
                    else:
                        trainer_opt.zero_grad()
            except RuntimeError as e:
                logger.info('RuntimeError Occured at %s' % train_names_list)
                logger.info('error: %s' % e)
                continue
            except BaseException as e:
                logger.info('Other Exception Occured at %s' % train_names_list)
                logger.info('error: %s' % e)
                continue
            
            if xconfig.use_bmuf and ((ibatch % xconfig.epoch_batch + 1) % xconfig.bmuf_params["sync_step"] == 0 or (ibatch+1) % xconfig.epoch_batch == 0):
                bmuf_update(trainer_model, global_models, momentums)
            batch_end_params = BatchEndParam(epoch=epoch, nbatch=ibatch % xconfig.epoch_batch + 1, model=trainer_model, locals=locals())
            speed_metric(batch_end_params)
            if val_scheduler_dict["use_warmup"] and epoch == 0:
                warmup_scheduler(batch_end_params)
            ibatch += 1
            
            # ibatch = xconfig.epoch_batch # 测试validate
            if ibatch == 4000 and global_rank == 0:
                save_pytorch_model(trainer_model, xconfig.model_prefix, epoch)
            if ibatch % xconfig.epoch_batch == 0:
                if global_rank == 0:
                    save_pytorch_model(trainer_model, xconfig.model_prefix, epoch)
                valid_name_values, valid_msg, valid_value = validate(trainer_model, valid_loader, xconfig.val_epoch_batch, val_scheduler_dict["valid_metric"])
                logger.info(valid_msg, epoch, *valid_name_values)
                toc = time.time()
                logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))
                king_scheduler(valid_value)
                epoch += 1
                ibatch = 0
                tic = time.time()
        pass
        #train_set.reset()


def init_processes(global_rank, local_rank, world_size, fn, backend='nccl'):
    """Initialze the distributed environment"""
    dist.init_process_group(backend, rank=global_rank, world_size=world_size)
    fn(global_rank, local_rank, world_size)


if __name__ == '__main__':
    if xconfig.auto_load_epoch is True:
        xconfig.load_epoch = check_load_epoch(xconfig.model_prefix)
        logger.info("set load epoch = {}".format(xconfig.load_epoch))
    logger.info(rain.xconfig.get_config_str())
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logger.info("World size: %d" % world_size)
    init_processes(global_rank, local_rank, world_size, train, backend='gloo')

