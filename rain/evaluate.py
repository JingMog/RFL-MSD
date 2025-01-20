import logging
logger = logging.getLogger()
from collections import namedtuple
import time
import numpy

BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'model',
                            'locals'])

class Speedometer(object):
    def __init__(self, batch_size, epoch_batch, frequent=50, opt=None, auto_reset=True):
        self.batch_size = batch_size
        self.epoch_batch = epoch_batch
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset

        self.opt = opt

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                name_values = []
                msg = 'Epoch[%d]\tBatch[%d][%d]\tlr[%f]\tSpeed: %.2f samples/sec'
                if param.model._eval_metrics is not None:
                    for i, eval_metric in enumerate(param.model._eval_metrics):
                        name_value = eval_metric.get()
                        name_values += name_value
                        if self.auto_reset:
                            eval_metric.reset()
                        msg += '\t%s=%f'
                    

                    if self.opt is None:
                        logging.info(msg, param.epoch, count, self.epoch_batch, 1, speed, *name_values)
                    else:
                        logging.info(msg, param.epoch, count, self.epoch_batch, self.opt.param_groups[0]['lr'], speed, *name_values)
                        
                else:
                    logging.info("Iter[%d]\tBatch[%d][%d]\tSpeed: %.2f samples/sec",
                                 param.epoch, count, self.epoch_batch, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


class WarmupScheduler(object):
    def __init__(self, optimizer, start_lr=1e-8, stop_lr=2e-4, step=10000, frequent=50):
        super(WarmupScheduler, self).__init__()
        self.optimizer   = optimizer
        self.start_lr    = start_lr
        self.stop_lr     = stop_lr
        self.step        = float(step)
        self.count       = 0.
        self.frequent    = frequent

    def __call__(self, param):
        if self.count < self.step and self.start_lr < self.stop_lr:
            self.count += 1
            next_lr = (self.count/self.step)*(self.stop_lr-self.start_lr) + self.start_lr
            self.optimizer.param_groups[0]['lr'] = next_lr
            if self.count % self.frequent == 0:
                logging.info('warmup[%d/%d]\tnext batch lr=%.2e' % (self.count,int(self.step),next_lr))


class KingScheduler(object):
    def __init__(self, optimizer, scheduler_dict):
        super(KingScheduler, self).__init__()
        self.optimizer    = optimizer
        self.scheduler    = scheduler_dict["scheduler"]
        self.lr_factor    = scheduler_dict["lr_factor"]
        self.wd_factor    = scheduler_dict["wd_factor"]
        self.eps_factor   = scheduler_dict["eps_factor"]
        self.stop_lr      = scheduler_dict["stop_lr"]
        self.stop_wd      = scheduler_dict["stop_wd"]
        self.stop_eps     = scheduler_dict["stop_eps"]
        self.decay_wd     = scheduler_dict["decay_wd"]
        self.decay_eps    = scheduler_dict["decay_eps"]
        self.thresh       = scheduler_dict["thresh"]
        self.decay_step   = scheduler_dict["decay_step"]
        self.cur_step     = 0
        self.cur_step_ind = 0
        self.descent      = -1.0 if scheduler_dict["valid_metric"] == 'ce' else 1.0 # ce or acc of validation
        self.max_val      = -numpy.inf
        
    def __call__(self, value=None):
        if self.scheduler == 'FixStep':
            self.cur_step_ind = 0
            self.cur_step += 1
        elif self.scheduler == 'AutoStep' and value is not None:
            self.cur_step_ind = 0
            value *= self.descent
            if (value - self.max_val) > self.thresh:
                self.cur_step = 0
                self.max_val  = value
            else:
                self.cur_step += 1
        elif self.scheduler == 'MultiStep':
            self.cur_step += 1

        # tune lr, eps, wd
        if self.cur_step == self.decay_step[self.cur_step_ind]:
            logging.info('{} {}'.format(self.scheduler, self.decay_step[self.cur_step_ind]))
            next_lr  = max(self.optimizer.param_groups[0]['lr']*self.lr_factor, self.stop_lr)
            next_wd  = max(self.optimizer.param_groups[0]['weight_decay']*self.wd_factor, self.stop_wd)
            self.optimizer.param_groups[0]['lr'] = next_lr
            logging.info('next epoch lr={}'.format(next_lr))
            if self.decay_wd:
                self.optimizer.param_groups[0]['weight_decay'] = next_wd 
                logging.info('next epoch wd={}'.format(next_wd))
            if 'eps' in self.optimizer.param_groups[0] and self.decay_eps:
                next_eps = max(self.optimizer.param_groups[0]['eps']*self.wd_factor, self.stop_wd)
                self.optimizer.param_groups[0]['eps'] = next_eps
                logging.info('next epoch eps={}'.format(next_eps))

            self.cur_step = 0
            self.cur_step_ind = min(self.cur_step_ind+1, len(self.decay_step)-1)
