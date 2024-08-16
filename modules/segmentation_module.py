
import math
from functools import partial, reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed
from torch.nn import init
import copy
import numpy as np





def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):
    def __init__(
        self,
        body,
        head,
        head_channels,
        classes,
        use_cosine=False,
        opts=None
    ):
        super(IncrementalSegmentationModule, self).__init__()
        self.body = body
        self.head = head

        if opts.dataset == "cityscapes_domain":
            classes = [opts.num_classes]
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)

        self.use_cosine = use_cosine

    def align_weight(self, align_type):
        old_weight_norm = self._compute_weights_norm(self.cls[:-1], only=align_type)

        new_weight_norm = self._compute_weights_norm(self.cls[-1:])

        gamma = old_weight_norm / new_weight_norm

        self.cls[-1].weight.data = gamma * self.cls[-1].weight.data

    def _compute_weights_norm(self, convs, only="all"):
        c = 0
        s = 0.

        for i, conv in enumerate(convs):
            w = conv.weight.data[..., 0, 0]

            if only == "old" and i == 0:
                w = w[1:]
            elif only == "background" and i == 0:
                w = w[:1]

            s += w.norm(dim=1).sum()
            c += w.shape[0]

        return s / c


    def init_new_classifier(self):
        imprinting_w = self.head.cls[0].weight[0]
        if not self.use_cosine:
            bkg_bias = self.head.cls[0].bias[0]

        if not self.use_cosine:
            bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).cuda()
            new_bias = (bkg_bias - bias_diff)

        self.head.cls[-1].weight.data.copy_(imprinting_w)
        if not self.use_cosine:
            self.head.cls[-1].bias.data.copy_(new_bias)

        if not self.use_cosine:
            self.head.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, mask=None, cls=None, prototypes=None, mode='get', weight_matrix=None):
        # mode: get,exist,no_proto,CIM
        out_size = x.shape[-2:]

        x_b, attentions = self.body(x)  # x_b是attention通过relu的输出
        out = self.head(x_b, mask, cls, proto_dict=prototypes, mode=mode, weight_matrix=weight_matrix)

        sem_logits = F.interpolate(
            out, size=out_size, mode="bilinear", align_corners=False
        )
        # elif phase == 'learn_base':
        #     sem_logits = self.NMS(sem_logits_small, out_size, cls)
        #


        return sem_logits

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def NMS(self, Q_mask, out_size, all_cls):
        cls_num = self.classes[0]
        batch_size = len(all_cls)

        out_masks = torch.zeros((batch_size, cls_num, out_size[0], out_size[1])).cuda()

        count = 0
        for b in range(batch_size):
            length = len(all_cls[b])

            if length == 1:
                out_mask_ = Q_mask[count].unsqueeze(0)
                out_mask_ = F.interpolate(
                    out_mask_, size=out_size, mode="bilinear", align_corners=False
                )
                # _, prediction = out_mask_.max(dim=1)
                # prediction = prediction.squeeze(0)
                out_masks[b][0] = out_mask_[0][0]
                out_masks[b][all_cls[b][0]] = out_mask_[0][1]
                # prediction[prediction == 1] = all_cls[b][0]
                # out_list.append(prediction.cpu().numpy().astype(int))
                count += 1
            else:
                for m in range(length):
                    v = Q_mask[count+m].unsqueeze(0)
                    v = F.interpolate(
                        v, size=out_size, mode="bilinear", align_corners=False
                    )

                    out_masks[b][0] += v[0][0] / length  # 平均
                    out_masks[b][all_cls[b][m]] = v[0][1]
                count += length


                # result = np.zeros(out_size)
                # for m in range(length):
                #     v = Q_mask[count+m].unsqueeze(0)
                #     v = F.interpolate(
                #         v, size=out_size, mode="bilinear", align_corners=False
                #     )
                #     v = F.softmax(v, dim=1)  # softmax
                #     x = v[0].detach().cpu().numpy()
                #     z = copy.deepcopy(x[1])  # foreground
                #     z[x[0] >= x[1]] = 0
                #     result[z > result] = all_cls[b][m]
                #
                # out_list.append(result.astype(int))
                # count += length
        assert count == Q_mask.size(0)
        # out = np.stack(out_list, axis=0)
        # out = torch.from_numpy(out).cuda()
        return out_masks

        # keys = Q_mask.keys()
        #
        # for k in keys:
        #     v = Q_mask[k]
        #     if type == 'large_first':
        #         v = F.interpolate(
        #             v, size=out_size, mode="bilinear", align_corners=False
        #         )
        #     v = F.softmax(v, dim=1)  # softmax
        #     x = v[0].cpu().numpy()
        #     z = copy.deepcopy(x[1])  # foreground
        #     z[x[0] >= x[1]] = 0
        #     result[z > result] = k
        #
        # result[x[0] < x[1]] = k
        # return result
            # print(f'\n after pred {k} has ', np.sum(result == 0), f' {0}\n')
            # for i in keys:
            #     print(f'after pred {k} has ', np.sum(result == i), f' {i}')