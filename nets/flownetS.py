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

class FlowNetS(Model):

    def __init__(self):
        super(FlowNetS, args).__init__(args)

        self.conv1 = nn.Conv2d(6, 64, 7, stride=2. padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2. padding=2)
        self.conv3 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)

        self.predict_flow6 = nn.Conv2d(1024,2,3,stride=1,padding=1, bias=False)
        self.predict_flow5 = nn.Conv2d(1026,2,3,stride=1,padding=1, bias=False)
        self.predict_flow4 = nn.Conv2d(770,2,3,stride=1,padding=1, bias=False)
        self.predict_flow3 = nn.Conv2d(386,2,3,stride=1,padding=1, bias=False)
        self.predict_flow2 = nn.Conv2d(194,2,3,stride=1,padding=1, bias=False)

        self.deconv5 = nn.ConvTranspose2d(1024,512,4,stride=2,padding=1)
        self.deconv4 = nn.ConvTranspose2d(1026,256,4,stride=2,padding=1)
        self.deconv3 = nn.ConvTranspose2d(770,128,4,stride=2,padding=1)
        self.deconv2 = nn.ConvTranspose2d(386,64,4,stride=2,padding=1)


        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2,2,4,2,1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2,2,4,2,1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2,2,4,2,1, bias=False)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2,2,4,2,1, bias=False)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2,2,4,2,1, bias=False)

    def _init_weights(self):
        pass

    def forward(self, image0_vb, image1_vb):
        conv = torch.cat((image0_vb, image1_vb), dim=1)
        conv1 = F.leaky_relu(self.conv1(conv),negative_slope=0.1, inplace=True)
        conv2 = F.leaky_relu(self.conv2(conv1),negative_slope=0.1, inplace=True)
        conv3 = F.leaky_relu(self.conv3(conv2),negative_slope=0.1, inplace=True)
        conv3 = F.leaky_relu(self.conv3_1(conv3),negative_slope=0.1, inplace=True)
        conv4 = F.leaky_relu(self.conv4(conv3),negative_slope=0.1, inplace=True)
        conv4 = F.leaky_relu(self.conv4_1(conv4),negative_slope=0.1, inplace=True)
        conv5 = F.leaky_relu(self.conv5(conv4),negative_slope=0.1, inplace=True)
        conv5 = F.leaky_relu(self.conv5_1(conv5),negative_slope=0.1, inplace=True)
        conv6 = F.leaky_relu(self.conv6(conv5),negative_slope=0.1, inplace=True)
        conv6 = F.leaky_relu(self.conv6_1(conv6),negative_slope=0.1, inplace=True)

        flow6 = self.predict_flow6(conv6)
        up_flow6 = self.upsampled_flow6_to_5(flow6)
        deconv5 = F.leaky_relu(self.deconv5(conv6),negative_slope=0.1,inplace=True)

        cat5 = torch.cat((conv5, deconv5, up_flow6),1) 
        flow5 = self.predict_flow5(cat5) 
        up_flow5 = self.upsampled_flow5_to_4(flow5) 
        deconv4 = F.leaky_relu(self.deconv4(cat5),negative_slope=0.1,inplace=True)

        cat4 = torch.cat((conv4, deconv4, up_flow5),1) 
        flow4 = self.predict_flow4(cat4) 
        up_flow4 = self.upsampled_flow4_to_3(flow4)
        deconv3 = F.leaky_relu(self.deconv3(cat4),negative_slope=0.1,inplace=True)

        cat3 = torch.cat((conv3, deconv3, up_flow4),1) 
        flow3 = self.predict_flow4(cat3) 
        up_flow3 = self.upsampled_flow4_to_3(flow3)
        deconv2 = F.leaky_relu(self.deconv2(cat3),negative_slope=0.1,inplace=True)

        cat2 = torch.cat((conv2, deconv2, up_flow3),1)
        flow2 = self.predict_flow2(cat2)

        return flow2
