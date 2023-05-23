#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 20:16:16 2022

@author: hal
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import cv2
import math
import weakref
import warnings
from functools import wraps

import os
import time

import torch.optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss


from ETL_torch import TissueDataset
from model_cast_torch import UDTransNet 

import ml_collections

def get_training_config():
    config_dict = ml_collections.ConfigDict()
    config_dict.kfold = 7
    config_dict.CosineLR = True
    config_dict.n_channels = 3
    config_dict.n_labels = 1
    config_dict.epochs = 50
    config_dict.print_frequency = 10
    config_dict.batch_size = 16
    config_dict.learning_rate = 1e-3
    config_dict.vis_path = './test_vis'
    
    return config_dict

def get_model_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 2
    config.transformer.embedding_channels = 32 * config.transformer.num_heads
    config.KV_size = config.transformer.embedding_channels * 4
    config.KV_size_S = config.transformer.embedding_channels
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0.1
    config.patch_sizes=[16,8,4,2]
    config.base_channel = 32
    config.decoder_channels = [32,64,128,256,512]
    return config

class WeightedBCE(nn.Module):

    def __init__(self, weights=[0.5, 0.5]):
        super(WeightedBCE, self).__init__()
        self.weights = weights

    def forward(self, logit_pixel, truth_pixel):
        logit = logit_pixel.view(-1)
        truth = truth_pixel.view(-1)
        assert(logit.shape==truth.shape)
        loss = torch.nn.functional.binary_cross_entropy(logit, truth, reduction='none')
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (self.weights[0]*pos*loss/pos_weight + self.weights[1]*neg*loss/neg_weight).sum()

        return loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.5]): # W_pos=0.8, W_neg=0.2
        super(WeightedDiceLoss, self).__init__()
        self.weights = weights

    def forward(self, logit, truth, smooth=1e-5):
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = truth.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        # p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
        # t = w*(t*2-1)
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)
        # print "------",dice.data

        loss = dice.mean()
        return loss

class BinaryDiceBCE(nn.Module):
    def __init__(self,dice_weight=1,BCE_weight=0):
        super(BinaryDiceBCE, self).__init__()
        self.BCE_loss = WeightedBCE(weights=[0.5, 0.5])
        self.dice_loss = WeightedDiceLoss(weights=[0.5, 0.5])
        self.BCE_weight = BCE_weight
        self.dice_weight = dice_weight

    def _show_dice(self, inputs, targets):
        inputs[inputs>=0.5] = 1
        inputs[inputs<0.5] = 0
        # print("2",np.sum(tmp))
        targets[targets>0] = 1
        targets[targets<=0] = 0
        hard_dice_coeff = 1.0 - self.dice_loss(inputs, targets)
        return hard_dice_coeff

    def forward(self, inputs, targets):
        # inputs = inputs.contiguous().view(-1)
        # targets = targets.contiguous().view(-1)
        # print "dice_loss", self.dice_loss(inputs, targets)
        # print "focal_loss", self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        BCE = self.BCE_loss(inputs, targets)
        # print "dice",dice
        # print "focal",focal
        dice_BCE_loss = self.dice_weight * dice + self.BCE_weight * BCE

        return dice_BCE_loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class MultiClassDiceCE(nn.Module):
    def __init__(self, num_classes=9):
        super(MultiClassDiceCE, self).__init__()
        self.CE_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)
        self.CE_weight = 0.5
        self.dice_weight = 0.5

    def _show_dice(self, inputs, targets, softmax=True):
        dice = 1.0 - self.dice_loss(inputs, targets,softmax=softmax)
        return dice

    def forward(self, inputs, targets, softmax=True):
        dice = self.dice_loss(inputs, targets,softmax=softmax)
        CE = self.CE_loss(inputs, targets)
        dice_CE_loss = self.dice_weight * dice + self.CE_weight * CE
        
        return dice_CE_loss
    

# =======================================================================
#      Learning Rate Scheduler: CosineAnnealingWarmRestarts
# =======================================================================
class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1):

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule. "
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
 

class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         scheduler.step(epoch + i / iters)
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
    
class TissueTrainer:
    def __init__(self):
        
        self.training_config = get_training_config()
        self.model_config = get_model_config()
        self.epochs = self.training_config.epochs
        self.kfolds = self.training_config.kfold
        self.batch_size = self.training_config.batch_size
        self.learning_rate = self.training_config.learning_rate
        self.curr_fold = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = TissueDataset(train_csv="./train.csv", img_size=224, batch_size=self.batch_size, debug=False)
        self.model = UDTransNet(config=self.model_config)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = BinaryDiceBCE() if self.training_config['n_labels'] == 1 else MultiClassDiceCE()
        self.lr_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=2, T_mult=2, eta_min=0) if self.training_config.CosineLR else None
        
        
    def dice_coef(self, y_true, y_pred):
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    
    def dice_on_batch(self, masks, pred):
        '''Computes the mean Area Under ROC Curve over a batch during training'''
        dices = []
    
        for i in range(pred.shape[0]):
            pred_tmp = pred[i][0].cpu().detach().numpy()
            # print("www",np.max(prediction), np.min(prediction))
            mask_tmp = masks[i].cpu().detach().numpy()
            pred_tmp[pred_tmp>=0.5] = 1
            pred_tmp[pred_tmp<0.5] = 0
            # print("2",np.sum(tmp))
            mask_tmp[mask_tmp>0] = 1
            mask_tmp[mask_tmp<=0] = 0
            # print("rrr",np.max(mask), np.min(mask))
            dices.append(self.dice_coef(mask_tmp, pred_tmp))
            
        return np.mean(dices)
    
    def save_on_batch(self, images1, masks, pred, names, vis_path):
        '''Computes the mean Area Under ROC Curve over a batch during training'''
        for i in range(pred.shape[0]):
            pred_tmp = pred[i][0].cpu().detach().numpy()
            mask_tmp = masks[i].cpu().detach().numpy()
            # img_tmp = images1[i].cpu().detach().numpy().reshape(224,224,3)
            # print(img_tmp.shape)
            pred_tmp[pred_tmp>=0.5] = 255
            pred_tmp[pred_tmp<0.5] = 0
            # print("2",pred_tmp.shape)
            mask_tmp[mask_tmp>0] = 255
            mask_tmp[mask_tmp<=0] = 0
    
            # img_tmp.save(vis_path+names[i][:-4]+".jpg")
            # cv2.imwrite(vis_path+names[i][:-4]+".jpg", img_tmp)
            cv2.imwrite(vis_path+ names[i][:-4]+"_pred.jpg", pred_tmp)
            cv2.imwrite(vis_path+names[i][:-4]+"_gt.jpg", mask_tmp)
            

    def print_summary(self, epoch, i, nb_batch, loss, fold,
                      average_loss, average_time,
                      dice, average_dice, mode, lr):
        '''
            mode = Train or Test
        '''
        summary = '   [' + str(mode) + '] Fold: [{0}/{1}] Epoch: [{2}][{3}/{4}]  '.format(fold, self.kfolds, epoch, i, nb_batch)
        string = ''
        string += 'Loss:{:.3f} '.format(loss)
        string += '(Avg {:.4f}) '.format(average_loss)
        string += 'Dice:{:.4f} '.format(dice)
        string += '(Avg {:.4f}) '.format(average_dice)
        if mode == 'Train':
            string += 'LR {:.2e}   '.format(lr)
        string += '(AvgTime {:.1f})   '.format(average_time)
        summary += string
        print(summary)
            
    def train_per_epoch(self, loader, epoch, fold, writer):
        logging_mode = 'Train'
        end = time.time()
        time_sum, loss_sum = 0, 0
        dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
        dices = []
        
        for (i, batch) in enumerate(loader):
            print(f"Epoch {epoch}: Batch {i}")
            images, masks = batch[0], batch[1]
            images, masks = images.cuda(device=self.device), masks.cuda(device=self.device)
            
            preds = self.model(images)
            if self.training_config.n_labels>1:
                out_loss = self.criterion(preds, masks.long(), softmax=True)
            else:
                out_loss = self.criterion(preds, masks.float())
            
            self.optimizer.zero_grad()
            out_loss.backward()
            self.optimizer.step()
            
        if self.training_config.n_labels>1:
            train_dice = self.criterion._show_dice(preds, masks.long(), softmax=True)
        else:
            train_dice = self.criterion._show_dice(preds, masks.float())
        
        batch_time = time.time() - end
        dices.append(train_dice)
        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        dice_sum += len(images) * train_dice
        
        if i == len(loader):
            average_loss = loss_sum / (self.training_config.batch_size*(i-1) + len(images))
            average_time = time_sum / (self.training_config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (self.training_config.batch_size*(i-1) + len(images))
            
        else:
            average_loss = loss_sum / (i * self.training_config.batch_size)
            average_time = time_sum / (i * self.training_config.batch_size)
            train_dice_avg = dice_sum / (i * self.training_config.batch_size)
        
        end = time.time()
        torch.cuda.empty_cache()
        
        if i % self.training_config.print_frequency == 0:
            self.print_summary(epoch=epoch+1, i=i, nb_batch=len(loader), loss=out_loss, fold=fold, 
                               average_loss=average_loss, average_time=average_time, dice=train_dice,
                               average_dice=train_dice_avg, mode=logging_mode, lr=min(g["lr"] for g in self.optimizer.param_groups))
            
        if self.training_config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(f"train_dice_fold_{fold}", train_dice, step)
        
        torch.cuda.empty_cache()
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        return average_loss, train_dice_avg
    
    def test_per_epoch(self, loader, epoch, fold, writer):
        logging_mode = 'Test'
        end = time.time()
        time_sum, loss_sum = 0, 0
        dice_sum, iou_sum, acc_sum = 0.0, 0.0, 0.0
        dices = []
        
        for (i, batch) in enumerate(loader):
            images, masks = batch[0], batch[1]
            images, masks = images.cuda(device=self.device), masks.cuda(device=self.device)
            names = batch[-1]
            preds = self.model(images)
            
            if self.training_config.n_labels>1:
                out_loss = self.criterion(preds, masks.long(), softmax=True)
            else:
                out_loss = self.criterion(preds, masks.float())
            

            
        if self.training_config.n_labels>1:
            train_dice = self.criterion._show_dice(preds, masks.long(), softmax=True)
        else:
            train_dice = self.criterion._show_dice(preds, masks.float())
        
        batch_time = time.time() - end
        
        if epoch % 10 == 0 and epoch !=0:
            vis_path = self.training_config.bis_path+"fold_"+str(fold)+"/"+str(epoch)+"/"
            if not os.path.isdir(vis_path):
                os.makedirs(vis_path)
            self.save_on_batch(images, masks, preds, names, vis_path)
        
        dices.append(train_dice)
        time_sum += len(images) * batch_time
        loss_sum += len(images) * out_loss
        dice_sum += len(images) * train_dice
        
        if i == len(loader):
            average_loss = loss_sum / (self.training_config.batch_size*(i-1) + len(images))
            average_time = time_sum / (self.training_config.batch_size*(i-1) + len(images))
            train_dice_avg = dice_sum / (self.training_config.batch_size*(i-1) + len(images))
            
        else:
            average_loss = loss_sum / (i * self.training_config.batch_size)
            average_time = time_sum / (i * self.training_config.batch_size)
            train_dice_avg = dice_sum / (i * self.training_config.batch_size)
        
        end = time.time()
        torch.cuda.empty_cache()
        

        
        if i % self.training_config.print_frequency == 0:
            self.print_summary(epoch=epoch+1, i=i, nb_batch=len(loader), loss=out_loss, fold=fold, 
                               average_loss=average_loss, average_time=average_time, dice=train_dice,
                               average_dice=train_dice_avg, mode=logging_mode, lr=min(g["lr"] for g in self.optimizer.param_groups))
            
        if self.training_config.tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(f"test_dice_fold_{fold}", train_dice, step)
        
        torch.cuda.empty_cache()
        
        return average_loss, train_dice_avg
    
    def save_checkpoint(self, state, save_path):
        '''
            Save the current model.
            If the model is the best model since beginning of the training
            it will be copy
        '''
        print('\t Saving to {}'.format(save_path))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        epoch = state['epoch']  # epoch no
        best_model = state['best_model']  # bool
        model = state['model']  # model type
        if best_model:
            filename = save_path + '/' + 'best_model-{}.pth.tar'.format(model)
        else:
            filename = save_path + '/' + 'model-{}-{:02d}.pth.tar'.format(model, epoch)
        torch.save(state, filename)
        
    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        this_fold = checkpoint['fold']
        return epoch, loss, this_fold
    
    def training_loop(self, loader_tup, fold, epochs, log_dir="./torch_tensorboard_logs"):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        
        writer = SummaryWriter(log_dir=log_dir)
        max_dice = 0.0
        best_epoch = 1
        model_name = "UDTRANS"
        load_path = f"./torch_models/best_model-{model_name}.pth.tar"
        if os.path.exists(load_path):
            new_epoch, loss, curr_fold = self.load_checkpoint(load_path)
            print("Best model from previous run found")
            self.curr_fold = curr_fold
            print(f"Beginning Traing at: {self.curr_fold}, epoch {new_epoch}")
        
        else:
            new_epoch = 0
            print("Starting from scratch")
        
        for epoch in range(new_epoch, self.epochs):
            print(f"\nTraining on Fold: {fold} Epoch: {epoch}")
            self.model.train(True)
            self.train_per_epoch(loader=loader_tup[0], epoch=epoch, fold=fold, writer=writer)
            
            with torch.no_grad():
                self.model.eval()
                val_loss, val_dice = self.test_per_epoch(loader=loader_tup[1], epoch=epoch, fold=fold, writer=writer)
                
            #Save Best model
            if val_dice > max_dice:
                if epoch + 1 > 1:
                    print('\t |||Saving best model, mean dice increased from: {:.4f} to {:.4f} ||||'.format(max_dice,val_dice))
                    
                    max_dice = val_dice
                    best_epoch = epoch + 1
                    checkpoint_state = {"epoch": epoch, "best_model": True,
                                         "state_dict": self.model.state_dict(),
                                         "model": model_name,
                                         "val_loss": val_loss,
                                         "optimizer": self.optimizer.state_dict(),
                                         "fold": fold
                                         }
                    
                    self.save_checkpoint(state=checkpoint_state, save_path="./torch_models")
                else: 
                    pass
                
            elif val_dice == 0:
                best_epoch = epoch + 1
                print("\t Reset count Number")
            
            else:
                print('\t Mean dice:{:.4f} does not increase, the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        
        return max_dice
    
    def train_kfold(self):
        print(f"\n--------------Beginning K-Fold Training of {self.kfolds} folds---------------------------\n")
        
        self.model = self.model.cuda(device=self.device)
        loader_tups = self.dataset.get_train_kfold()
        fold = 1
        
        for tup_idx in range(self.curr_fold, self.kfolds):
            
            fold_max_dice = self.training_loop(loader_tups[tup_idx], fold, epochs=self.epochs)
            print(f"\n~~~~|||Maximum Training Dice Accuracy for fold {fold}: {fold_max_dice}")
            fold = fold + 1
            
        torch.save(self.model, "final_torch_model.pt")
        
        print("\n------------------------Finished Training-------------------")
                
def train_model():
    trainer = TissueTrainer()
    trainer.train_kfold()
    
if __name__ == "__main__":
    train_model()
                    
                
        
        
    
    
        
            
            
            
                


            
        
            
            
    