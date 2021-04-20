
import types
import math
from torch._six import inf
from functools import wraps, partial
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim.optimizer import Optimizer
# from torch.optimizer import Optimizer

from torch.optim.lr_scheduler import _LRScheduler

# Paper : https://arxiv.org/pdf/1608.03983.pdf
# class _LRScheduler(object): 부모클래스도 커스텀해야하나 했지만 필요 X
# 기본 부모클래스 Scheduler가 last_lr을 get하는 라인을 사용하고 있어서 수정이 필요했음.

class CustomizedCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        # T_0 : 첫 cycle의 에폭수를 몇으로 할지
        # T_mult : 다음 주기를 이전의 주기보다 몇배로 할지 (int)
        # eta_max : learning rate의 상한선
        # T_up : warmup step을 몇으로 할지
        # gamma : 이전에서 다음 cycle로 넘어갈 때마다 max_lr을 줄여줍니다
        # min_lr은 여기서 따로 선언하지 않고, Optimizer 객체 생성시 지정해주는 lr이 그것에 해당됩니다.
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        # self.T_cur = last_epoch
        super(CustomizedCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = last_epoch
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        ####################################################################
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        ####################################################################
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

