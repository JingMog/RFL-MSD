import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import xconfig

class EvalMetric(object):
    """Base class for all evaluation metrics.
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(**config)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred
        """

        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_name_value(self):
        """Returns zipped name and value pairs.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))



class MyCrossEntropy(EvalMetric):
    def __init__(self, eps=1e-8, name='train'):
        super(MyCrossEntropy, self).__init__(name+"-ce")
        self.eps = eps

    @torch.no_grad()
    def update(self, labels, preds): # labels list  preds list
        label, target_hook, mask = (l.detach() for l in labels)
        pred = preds[3].detach() # 2521
        
        #label = label.T.flatten().astype('int32')
        #mask = mask.T.flatten()
        label = label.flatten().long()
        mask = mask.flatten()
        ce = pred[torch.arange(len(label)), label]
        ce = -torch.log(ce+self.eps) *mask
        #ce= ce.sum()/mask.sum()
       
        self.sum_metric += float(ce.sum().item())
        self.num_inst += mask.sum().item()
        
class MyACC(EvalMetric):
    def __init__(self, name="train"):
        super(MyACC, self).__init__(name+"-acc")

    @torch.no_grad()
    def update(self, labels, preds):
        label, target_hook, mask = (l.detach() for l in labels)
        pred = preds[3].detach()
        label = label.long()
        label = label.flatten()
        mask = mask.flatten()
        rec = torch.argmax(pred, axis=1)
        rec = rec.long()
        acc = rec== label
        acc= acc*mask
        #acc = acc.sum()/mask.sum()
        
        self.sum_metric += acc.sum().item()
        self.num_inst += mask.sum().item()

class MyLossCand(EvalMetric):
    def __init__(self, eps=1e-8, name='train'):
        super(MyLossCand, self).__init__(name+"-cand_loss")
        self.eps = eps

    def update(self, labels, preds): # labels list  preds list
        # label, target_cand_angle, target_hook, mask = (l.detach().cpu().numpy() for l in labels)
        loss = preds[1].detach().cpu().numpy() # [B, ]
        self.sum_metric += loss
        self.num_inst += 1

class MyLossMem(EvalMetric):
    def __init__(self, eps=1e-8, name='train'):
        super(MyLossMem, self).__init__(name+"-mem_loss")
        self.eps = eps

    def update(self, labels, preds): # labels list  preds list
        # label, target_cand_angle, target_hook, mask = (l.detach().cpu().numpy() for l in labels)
        if torch.isnan(preds[1]):
            # print("nan, skip")
            return
        loss = preds[1].detach().cpu().numpy() # [B, ]
        self.sum_metric += loss
        self.num_inst += 1

class MyAccMem(EvalMetric):
    def __init__(self, eps=1e-8, name='train'):
        super(MyAccMem, self).__init__(name+"-mem_acc")
        self.eps = eps
        self.ea_index = xconfig.vocab.getID("<ea>")

    @torch.no_grad()
    def update(self, labels, preds): # labels list  preds list
        # label, target_cand_angle, target_hook, mask = (l.detach().cpu().numpy() for l in labels)
        if True in torch.isnan(preds[4].detach()) or True in torch.isnan(preds[5].detach()) or True in torch.isnan(preds[5]):
            return
        pred = preds[4].detach() # [B, L, M, v]
        rec = torch.argmax(pred, axis=-1) #[B, L, M]
        mem_tgt = preds[5].detach()
        mask = preds[6].detach()
        acc = (rec == mem_tgt)
        acc = acc * mask
        self.sum_metric += acc.sum()
        self.num_inst += mask.sum()

def GenTranMetric(name="train"):
    return [MyCrossEntropy(name=name), MyACC(name=name), MyLossMem(name=name), MyAccMem(name=name)]