import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import pickle
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import os
import random


class MetaLearner(nn.Module):
    def __init__(
            self,
            body,
            head,
            opts=None):
        super(MetaLearner, self).__init__()
        self.body = body
        self.head = head
        self.opts = opts

        self.scalar = None

    def forward(self, data_query, labels_query, labels_base, labels_novel, proto_dict, meta_test=False):

        if not meta_test:
            with torch.no_grad():
                bad_prototype_list = []
                for i in range(len(data_query)):
                    data_query_ = data_query[i]
                    fea_novel_, _ = self.body(data_query_.reshape([-1] + list(data_query_.shape[-3:])))
                    proto_cls = self.get_proto_img(fea_novel_, labels_query[i], labels_novel[i])
                    bad_prototype_list.append(proto_cls.mean(dim=0).unsqueeze(0))

                bad_prototypes = torch.cat(bad_prototype_list, dim=0)

                proto_base = torch.cat(
                    [proto_dict[cls] for cls in labels_base], dim=0).cuda()
                proto_novel = torch.cat(
                    [proto_dict[cls] for cls in labels_novel], dim=0).cuda()

            boost_prototypes = self.head(proto_base, bad_prototypes)
            return boost_prototypes, bad_prototypes, proto_novel
        else:
            with torch.no_grad():
                bad_prototype_list = []
                for i in range(len(data_query)):
                    data_query_ = data_query[i]
                    fea_novel_, _ = self.body(data_query_.reshape([-1] + list(data_query_.shape[-3:])))
                    proto_cls = self.get_proto_img(fea_novel_, labels_query[i], labels_novel[i])
                    bad_prototype_list.append(proto_cls.mean(dim=0).unsqueeze(0))

                bad_prototypes = torch.cat(bad_prototype_list, dim=0)

            boost_list = []
            for i in range(10):
                labels_base_no_zero = self.__strip_zero(labels_base)
                temp_labels_base = random.sample(labels_base_no_zero, self.opts.label_base_num)
                proto_base = torch.cat(
                    [proto_dict[cls] for cls in temp_labels_base], dim=0).cuda()

                boost_prototypes_ = self.head(proto_base, bad_prototypes)
                boost_list.append(boost_prototypes_)
            boost_prototypes = torch.stack(boost_list, dim=0).mean(dim=0)
            return boost_prototypes, bad_prototypes, None

    def get_proto_img(self, x, mask, cls, method='map'):
        feature_size = x.shape[2:]
        mask = mask.unsqueeze(1).to(torch.float32)
        mask[mask != cls] = 0
        mask[mask == cls] = 1
        mask = F.interpolate(mask, feature_size, mode='bilinear', align_corners=True)

        feature1 = x * mask
        if method == 'FPM':
            proto_img = self.FPM(feature1)
        elif method == 'map':
            b, c, h, w = feature1.size()
            feature1 = feature1.view(b, c, h * w)  # b * c * n
            proto_img = torch.mean(feature1, dim=2)

        return proto_img

    def __strip_zero(self, labels):
        while 0 in labels:
            labels.remove(0)
        return labels

class ProtoComNet(nn.Module):
    def __init__(self, in_dim=None, opts=None):
        super(ProtoComNet, self).__init__()

        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.in_dim, out_features=self.in_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.in_dim // 2, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=self.in_dim)
        )
        self.aggregator = nn.Sequential(
            nn.Linear(in_features=600 + 512, out_features=300),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=300, out_features=1)
        )
        self.adj_base = torch.randn(opts.label_novel_num, opts.label_base_num, requires_grad=True).cuda()  # 1*19
        self.adj_novel = torch.ones(opts.label_novel_num, opts.label_novel_num).cuda()

    def forward(self, proto_base, bad_prototypes):
        n_base = proto_base.size(0)
        n_novel = bad_prototypes.size(0)
        d = bad_prototypes.size(1)

        # 基于proto_base 去更新bad_prototypes
        input_feature = torch.cat([proto_base, bad_prototypes], dim=0)
        z = self.encoder(input_feature)
        z_base = z[:n_base]
        z_novel = z[n_base:]

        g_base = torch.mm(self.adj_base, z_base)
        g_novel = torch.mm(self.adj_novel, z_novel)
        g = g_base + g_novel
        boost_prototypes = self.decoder(g)

        return boost_prototypes

    def reparameterize(self, mu, var):
        std = var
        eps = torch.randn_like(std)
        return mu + eps * std
