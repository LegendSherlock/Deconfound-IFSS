# -*- coding: utf-8 -*-


import torch as t
import torch.nn.functional as F
import torch.nn as nn

class FPM(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        inplace = True
        conv_out = 256
        # reduce layers
        self.a1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.a2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.a3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            # nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(128, 128, kernel_size=7, stride=1)
        self.fc = nn.Linear(128*20*20, 1024)

    def forward(self, x):

        # Top-down
        aa1 = self.a1(x)
        aa2 = self.a2(aa1)
        aa3 = self.a3(aa2)

        bb1 = self.b1(x) + aa1
        bb2 = self.b2(bb1) + aa2
        bb3 = self.b3(bb2) + aa3
        out = self.out_conv(bb3)
        out = out.view(-1, 128*20*20)
        out = self.fc(out)
        return out
