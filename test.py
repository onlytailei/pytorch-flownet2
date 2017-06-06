#!/usr/bin/env python
# coding=utf-8
'''
Author:Tai Lei
Date:Mi 07 Jun 2017 00:45:32 CEST
Info:
'''

import torch
from nets.flownetC import CorrelationLayer
from torch.autograd import Variable

a = Variable(torch.randn(2,3,20,30))
b = Variable(torch.randn(2,3,20,30))
test_layer = CorrelationLayer()


print test_layer(a,b)
