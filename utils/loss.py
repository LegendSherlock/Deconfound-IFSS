import torch
import torch.nn as nn
import torch.nn.functional as F




class UnbiasedCrossEntropy(nn.Module):

    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets, mask=None):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        # Following line was fixed more recently in:
        # https://github.com/fcdl94/MiB/commit/1c589833ce5c1a7446469d4602ceab2cdeac1b0e
        # and added to my repo the 04 August 2020 at 10PM
        labels = targets.clone()  # B, H, W

        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        if mask is not None:
            labels[mask] = self.ignore_index
        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss



class KnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1., kd_cil_weights=False):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.kd_cil_weights = kd_cil_weights

    def forward(self, inputs, targets, mask=None):
        inputs = inputs.narrow(1, 0, targets.shape[1])

        outputs = torch.log_softmax(inputs, dim=1)
        labels = torch.softmax(targets * self.alpha, dim=1)

        loss = (outputs * labels).mean(dim=1)
        if self.kd_cil_weights:
            w = -(torch.softmax(targets, dim=1) * torch.log_softmax(targets, dim=1)).sum(dim=1) + 1.0
            loss = loss * w[:, None]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedKnowledgeDistillationLoss(nn.Module):

    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(
            inputs.device
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(
            torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1
        ) - den  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg +
                (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if self.reduction == 'mean':
            outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
