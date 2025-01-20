import os
from rain.utils import Vocab
import logging
logger = logging.getLogger()

# # data path
source_dim = 3
level = 'line' # line column topic
dst_type_dict={'ContentType': ['text'], 'LogicalType':['answer', 'subject'], 'PhysicalType': ['hand', 'print'], 'ImageType': ['normal']}

# train dataset
train_lrc = "ssml_train.lrc"
train_key  = train_lrc + ".line.cache"
train_key_inds = ()
train_ignore = None

# devdataset
dev_lrc = "ssml_valid.lrc"
dev_key    = dev_lrc + ".line.cache"

# test dataset
test_lrc = "ssml_valid.lrc"
test_lrc_cache = test_lrc + ".line.cache"
test_lrc_normh = 40

# amp setting
train_amp = False

# # aug setting
# rand resize
rand_resize = True
rand_resize_ratio = (0.6, 1.2)
rand_crop = True
rand_crop_pixel   = 6
# coutout
do_cutout = False
min_contour_area = 100.
width_rate =1.3
do_cut_rate =0.5
cutout_sample_rate = 0.5 
ignore_pixel =50 
# blur
do_blur = True
# rand polygon
rand_bbox_rate = 0.5
# random_scale_downup
do_random_scale_downup=False
random_scale_downup_range=(0.4, 0.75)

# # vocab params
vocab_file = "./dict/vocab.txt"
vocab      = Vocab(vocab_file, unk_id=0)
vocab_size = vocab.getVocSize()
sos        = vocab.get_sos()
eos        = vocab.get_eos()
enter      = vocab.getID("\\enter")

# # model params
base_model_dir = "" # model save path
base_model_dir = base_model_dir
model_prefix = base_model_dir + "/encdec"
model_type   = "pytorch"
num_epochs   = 100

# # train data params
max_height     = 2000
max_width      = 2000
max_length     = 10000
max_image_size = 6000000 #6000000
max_batch_size = 8
fix_batch_size = None

# # test data params
test_max_height     = 1000
test_max_width      = 1000
test_max_length     = 10000
test_fix_batch_size = 1
test_image_list     = None
test_image_normh    = 40
test_load_epochs    = '85'
test_key = ''
test_lrc = ''


img_fix_char_height = None
test_det_sections   = None
# ============================ Phase Params ========================== #
# # Train phase params
learning_rate      = 2e-4
weight_decay       = 0
seed               = 369
disp_batches       = 100
auto_load_epoch    = False
load_epoch         = None # if None, model params for training will be initialized randomly
load_param_path    = None
allow_missing      = True
epoch_batch        = 1
data_divide_num    = 0.2 # 5 data as one epoch
val_epoch_batch    = 1000
val_scheduler_dict = {'scheduler':'MultiStep', 'valid_metric': 'ce', # ce or acc
                      'lr_factor': 0.5, 'wd_factor':0.1, 'eps_factor': 0.1, 
                      'stop_lr': 1e-8,'stop_wd': 1e-12, 'stop_eps': 1e-12,
                      'decay_wd': False, 'decay_eps': False, 'thresh': 1e-5,
                      'decay_step': [12, 6, 3, 2, 1, 1, 1, 1, 1], # [40, 20, 10, 5, 1, 1, 1, 1, 1] #[125, 20, 10, 5, 1, 1, 1, 1, 1]
                       # warmup params
                      'use_warmup': True, 'warmup_start_lr': 1e-8,
                      'warmup_step': 1000, 'warmup_disp_freq': 50} # MultiStep decay_step, AutoStep decay_step[0], FixStep decay_step[0]
use_bmuf           = True
bmuf_params        = {"sync_step":50, "alpha":1, "blr":1.0, "bm": 0.875}
if use_bmuf:
    num_gpus = 1
    if 'WORLD_SIZE' in os.environ:
        num_gpus = int(os.environ['WORLD_SIZE'])
    epoch_batch = int(1 / (num_gpus*data_divide_num)) # 16000 batch as one epoch
    #epoch_batch = 64000 // num_gpus # 16000 batch as one epoch
    if num_gpus == 4:
        bmuf_params["sync_step"] = 50
        bmuf_params["bm"] = 0.75
    elif num_gpus == 8:
        bmuf_params["sync_step"] = 50
        bmuf_params["bm"] = 0.875
    elif num_gpus == 12:
        bmuf_params["sync_step"] = 50
        bmuf_params["bm"] = 0.8875
    elif num_gpus == 16:
        bmuf_params["sync_step"] = 50
        bmuf_params["bm"] = 0.9
    elif num_gpus == 32:
        bmuf_params["sync_step"] = 25
        bmuf_params["bm"] = 0.9
    elif num_gpus == 1:
        logger.info("Gpu count = 1 means that single card debug mode or test mode was launched")
    else:
        bmuf_params["sync_step"] = 50
        bmuf_params["bm"] = 0.75
        #raise ValueError("Gpu count = %d error, which should be in [4,8,16,32] if use bmuf" % num_gpus)

# # test phase params
frame_per_char   = 50
beam             = 5

# ========================== Encoder Params ========================== #
# # VGG16
encoder_units            = [3, 4, 6, 3]
encoder_use_res          = [1, 1, 1, 1]
encoder_basic_group      = [8, 16, 16, 32]
encoder_filter_list      = [24, 48, 96, 192, 384]
encoder_stride_list      = [(2, 2), (2, 2), (2, 2), (2, 2)]
encode_dropout           = 0.00
encode_feat_dropout      = 0.05

# # SelfAtten structure
encoder_position_dim    = 384
encoder_position_att    = 192
encoder_dim             = encoder_position_dim


# ========================== Decoder Params ========================== #
decoder_state_dim     = 256
decoder_embed_dim     = 128
decoder_att_dim       = 128
decoder_merge_dim     = 384
decoder_chatt_dim     = 384 # group channel attention
decoder_max_seq_len   = 1000000
decoder_dropout       = 0.2
decoder_embed_drop    = 0.15
decoder_cover_kernel  = (11,11)
decoder_cover_padding = (5,5)

decoder_angle_embed_dim = 128
decoder_mem_match_dim = 256

# =========================== Other Params ========================== #
def get_config_str():
    res = ''
    res += 'Config:\n'
    import collections
    hehe = collections.OrderedDict(sorted(globals().items(), key=lambda x: x[0]))
    for k, v in hehe.items():
        if k.startswith('__'): continue
        if k.startswith('SEPARATOR'): continue
        if k.startswith('get'): continue
        if type(v) == (type(os)): continue
        if len(k) < 2: continue
        res += '{0}: {1}\n'.format(k, v)
    return res
