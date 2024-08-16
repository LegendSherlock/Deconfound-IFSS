
import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.modules.FPM import FPM
import numpy as np
from modules.modules.deeplab import DeeplabV3


class deeplab_with_proto(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=256,
                 out_stride=16,
                 norm=nn.BatchNorm2d,
                 pooling_size=None,
                 classes=None):
        super(deeplab_with_proto, self).__init__()

        self.classes = classes
        self.FPM = FPM(in_channels)
        self.proto_list = []
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=2*in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=2, dilation=2,
                      bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels+2, out_channels=in_channels, kernel_size=3, stride=1, padding=1,
                      bias=True),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.seg_model = DeeplabV3(in_channels,
                 out_channels,
                 hidden_channels=hidden_channels,
                 out_stride=out_stride,
                 norm=norm,
                 pooling_size=pooling_size)

        self.cls = nn.ModuleList([nn.Conv2d(out_channels, c, 1, bias=True) for c in classes])
        # self.cls_2 = nn.Conv2d(out_channels, 2, 1, bias=True)

    def forward(self, x, all_mask, all_cls, proto_dict=None, mode='get', weight_matrix=None):
        if mode == 'get':
            proto_cls_list = []
            for cls in range(self.classes[0]):
                cls_mask = all_mask.clone()
                cls_mask[cls_mask != cls] = 0
                proto_cls = self.get_proto_img(x, cls_mask)
                proto_cls = proto_cls.unsqueeze(2).unsqueeze(3)
                proto_cls = proto_cls.expand(x.shape[0], -1, x.shape[2], x.shape[3])
                proto_cls_list.append(proto_cls)
            for p_c in proto_cls_list:
                fusion_feature = torch.add(x, p_c)

            pred_pl = self.seg_model(fusion_feature)

            out = []
            for i, mod in enumerate(self.cls):
                out.append(mod(pred_pl))
            pred_mask = torch.cat(out, dim=1)
            return pred_mask
        elif mode == 'exist':
            fusion_feature_list = []
            for b in range(len(all_cls)):
                x_b = x[b].unsqueeze(0)
                proto_list = []
                for cls_name in all_cls[b]:
                    proto_cls = proto_dict[cls_name].cuda()
                    proto_cls = proto_cls.unsqueeze(2).unsqueeze(3)
                    proto_cls = proto_cls.expand(x_b.shape[0], -1, x_b.shape[2], x_b.shape[3])
                    proto_list.append(proto_cls)

                fusion_feature_b = None
                for p_c in proto_list:
                    fusion_feature_b = torch.add(x_b, p_c)
                if fusion_feature_b is not None:
                    fusion_feature_list.append(fusion_feature_b)
                else:
                    fusion_feature_list.append(x_b)


            fusion_feature = torch.cat(fusion_feature_list, dim=0)
            pred_pl = self.seg_model(fusion_feature)

            out = []
            for i, mod in enumerate(self.cls):
                out.append(mod(pred_pl))
            pred_mask = torch.cat(out, dim=1)
            return pred_mask
        elif mode == 'no_proto':
            pred_pl = self.seg_model(x)

            out = []
            for i, mod in enumerate(self.cls):
                out.append(mod(pred_pl))
            pred_mask = torch.cat(out, dim=1)
            return pred_mask
        elif mode == 'CIM':
            pred_mask_proto_list = []
            for cls in range(1, self.classes[0]):
                proto_cls = proto_dict[cls].cuda()
                proto_cls = proto_cls.unsqueeze(2).unsqueeze(3)
                proto_cls = proto_cls.expand(x.shape[0], -1, x.shape[2], x.shape[3])
                fusion_feature = torch.add(x, proto_cls)
                pred_pl = self.seg_model(fusion_feature)

                out = []
                for i, mod in enumerate(self.cls):
                    out.append(mod(pred_pl))
                pred_mask_proto = torch.cat(out, dim=1)

                multi_vector = torch.cat([torch.ones((1,)).cuda(), weight_matrix[cls-1]], dim=0).view(1,-1,1,1).cuda()
                pred_mask_proto_out = pred_mask_proto * multi_vector
                pred_mask_proto_list.append(pred_mask_proto_out)

            # NMS
            pred_mask = self.NMS_CIM(pred_mask_proto_list)

            return pred_mask
        # else:
        #     # iter
        #     iter_num = 3
        #
        #     # feature concate
        #     b, c, w, h = x.size()
        #     Pseudo_mask = (torch.zeros(b, 2, w, h)).cuda()
        #
        #     out_list = []
        #     out_pl_list = []
        #
        #     for iter in range(iter_num):
        #         proto_img = self.get_proto_img(x, all_mask)
        #         proto_img = proto_img.unsqueeze(2).unsqueeze(3)
        #         proto_img = proto_img.expand(-1, -1, x.shape[2], x.shape[3])
        #         fusion_feature1 = self.fusion(torch.cat([x, proto_img], dim=1))
        #         fusion_feature2 = self.fusion2(torch.cat([fusion_feature1, Pseudo_mask], dim=1))
        #
        #         pred_pl = self.seg_model(fusion_feature2)
        #         out_pl_list.append(pred_pl)
        #
        #         pred_mask = self.cls_2(pred_pl)
        #         out_list.append(pred_mask)
        #
        #         pred_softmax = F.softmax(pred_mask, dim=1)
        #         Pseudo_mask = pred_softmax
        #
        #     pred_pl = torch.stack(out_pl_list, dim=0).mean(dim=0)
        # return pred_mask, pred_pl

        # elif self.mode == 'iter':
        #     # iter
        #     iter_num = 3
        #
        #     # feature concate
        #     b, c, w, h = x.size()
        #     Pseudo_mask = (torch.zeros(b, 2, 50, 50)).cuda()
        #     out_list = []
        #
        #     for iter in range(iter_num):
        #         out_pl_list = []
        #         for cls in range(1, self.classes[0]):  # 为每个类(不包括背景)计算原型并输出预测mask
        #             proto_cls = proto_dict[cls].cuda()
        #             proto_cls = proto_cls.unsqueeze(2).unsqueeze(3)
        #             proto_cls = proto_cls.expand(x.shape[0], -1, x.shape[2], x.shape[3])
        #
        #             fusion_feature1 = self.fusion(torch.cat([x, proto_cls], dim=1))
        #             fusion_feature2 = self.fusion2(torch.cat([fusion_feature1, Pseudo_mask], dim=1))
        #
        #             after_head = self.seg_model(fusion_feature2)
        #
        #             out_pl_list.append(after_head)
        #
        #         pred_pl = torch.cat(out_pl_list, dim=1)
        #         out = []
        #         out_softmax = []
        #         for i, mod in enumerate(self.cls):
        #             after_mod = mod(pred_pl)
        #             out.append(after_mod)
        #             out_softmax.append(F.softmax(after_mod, dim=1))
        #
        #         pred_mask = torch.cat(out, dim=1)
        #         Pseudo_mask = torch.cat(out_softmax, dim=1)
        #
        #         out_list.append(pred_mask)
        #
        #
        #         pred_pl = torch.stack(out_pl_list, dim=0).mean(dim=0)
        # return pred_mask, pred_pl

    def get_proto_img(self, x, mask, method='map'):
        feature_size = x.shape[2:]
        mask = mask.unsqueeze(1).to(torch.float32)
        mask = F.interpolate(mask, feature_size, mode='bilinear', align_corners=True)
        feature1 = x * mask
        if method=='FPM':
            proto_img = self.FPM(feature1)
        elif method=='map':
            b, c, h, w = feature1.size()
            feature1 = feature1.view(b, c, h * w)  # b * c * n
            proto_img = torch.mean(feature1, dim=2)

        return proto_img

    def NMS_CIM(self, mask_list):
        # 即选择概率最大的mask
        b, c, h, w = mask_list[0].size()
        result = torch.zeros((b, self.classes[0], h, w)).cuda()

        for i in range(b):
            for j in range(c):
                cls_mask_list = []
                for mask in mask_list:
                    x = mask[i][j].view(1, h*w)
                    cls_mask_list.append(x)
                cls_masks = torch.cat(cls_mask_list, dim=0)
                cls_mask = torch.max(cls_masks, dim=0).values.view(h, w)

                result[i,j, :] = cls_mask
        return result
