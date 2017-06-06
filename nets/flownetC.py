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
from model import Model

class FlowNetC(Model):

    def __init__(self, args):
        
        super(FlowNetC,self).__init__(args)
        self.conv1a = nn.Conv2d(3, 32, 7, stride=2. padding=3)
        self.conv1b = nn.Conv2d(3, 32, 7, stride=2. padding=3)
        self.conv2a = nn.Conv2d(32, 64, 5, stride=2. padding=2)
        self.conv2b = nn.Conv2d(32, 64, 5, stride=2. padding=2)
        self.conv3a = nn.Conv2d(64, 128, 5, stride=2. padding=2)
        self.conv3b = nn.Conv2d(64, 127, 5, stride=2. padding=2)

    def _init_weights(self):
        pass

    def forward(self, image0, image1):
        pass


