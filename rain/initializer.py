import numpy as np
import logging
import os, sys
logger = logging.getLogger()
import torch

def initialize_model(model, model_prefix, model_type='pytorch', epoch=None, rank=0, allow_missing=False):
    model_dir = "/".join(model_prefix.split("/")[:-1])
    if rank == 0 and not os.path.exists(model_dir):
        os.mkdir(model_dir)

    initialize_model_from_pytorch(model, model_prefix, epoch, allow_missing)

def initialize_model_from_pytorch(model, model_prefix, epoch=None, allow_missing=False):
    if epoch is not None:
        model_path = '%s-%04d.pt'%(model_prefix, epoch)
        logger.info("Loading params from %s" % model_path)
        if not os.path.exists(model_path):
            model_path = model_path.replace("_for_test", "")
        param_state_dict = torch.load(model_path, map_location='cpu')
        model_state_dict = model.state_dict()
        if allow_missing:
            for k, v in param_state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    model_state_dict[k] = v
                else:
                    logger.info("param %s can't be loaded, which shape is %s" % (k, v.shape))
            model.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(param_state_dict, strict = False)

def initialize_model_from_pytorch_v2(model, model_path, allow_missing=False):
    if model_path is not None:
        logger.info("Loading params from %s" % model_path)
        param_state_dict = torch.load(model_path, map_location='cpu')
        if "net" in param_state_dict:
            param_state_dict = param_state_dict["net"]
        model_state_dict = model.state_dict()
        if allow_missing:
            for k, v in param_state_dict.items():
                if k.find("decoder") != -1:
                    continue
                if k in model_state_dict and model_state_dict[k].shape == v.shape:
                    model_state_dict[k] = v
                else:
                    logger.info("param %s can't be loaded, which shape is %s" % (k, v.shape))
            model.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(param_state_dict, strict = False)

def save_pytorch_model(model, model_prefix, epoch):
    torch.save(model.state_dict(), '%s-%04d.pt'%(model_prefix, epoch))