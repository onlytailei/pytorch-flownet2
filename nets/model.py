#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:So 04 Jun 2017 22:39:19 CEST
Info: flownet model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        #self.logger = args.logger

    def _init_weights(self):
        raise NotImplementedError("not implemented in base calss")

    def print_model(self):
        self.logger.warning("<-----------------------------------> Model")
        self.logger.warning(self)

    def _reset(self):           # NOTE: should be called at each child's __init__
        self._init_weights()
        self.type(self.dtype)   # put on gpu if possible
        self.print_model()

    def forward(self, input):
        raise NotImplementedError("not implemented in base calss")

