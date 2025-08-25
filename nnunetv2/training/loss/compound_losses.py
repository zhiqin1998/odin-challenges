from collections import namedtuple

import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss, FocalLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param focal_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_focal:
        :param weight_dice:
        """
        super(DC_and_Focal_loss, self).__init__()

        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        self.focal = FocalLoss(**focal_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        dc_loss = self.dc(net_output, target, loss_mask=None) \
            if self.weight_dice != 0 else 0
        ce_loss = self.focal(net_output, target) \
            if self.weight_focal != 0 else 0

        result = self.weight_focal * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_CE_smooth_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss, smooth_mapping=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_smooth_loss, self).__init__()
        if ignore_label is not None:
            raise NotImplementedError

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        if smooth_mapping is None: # default for single pulp
            def smooth_tensor(cls_id, smooth_ids, smooth=0.1):
                if not isinstance(smooth_ids, list):
                    smooth_ids = [smooth_ids]
                temp = torch.zeros(47, device='cuda')
                temp[smooth_ids] = smooth / len(smooth_ids)
                temp[cls_id] = 1 - smooth
                return temp
            smooth_mapping = {3: smooth_tensor(3, 43), 4: smooth_tensor(4, 44),
                              5: smooth_tensor(5, 6), 6: smooth_tensor(6, 5),
                              11: smooth_tensor(11, [12, 19, 46]), 19: smooth_tensor(19, [11, 20, 46]),
                              27: smooth_tensor(27, [28, 35, 46]), 35: smooth_tensor(35, [27, 36, 46]),
                              43: smooth_tensor(43, 3), 44: smooth_tensor(44, 4),
                              }
            for i in list(range(12, 18)) + list(range(20, 26)) + list(range(28, 34)) + list(range(36, 42)):
                smooth_mapping[i] = smooth_tensor(i, [i-1, i+1, 46])
            for i in [18, 26, 34, 42]:
                smooth_mapping[i] = smooth_tensor(i, [i-1, 46])
        self.smooth_mapping = smooth_mapping
        onehot_tensor = torch.eye(len(next(iter(self.smooth_mapping.values()))), device='cuda')
        for k, v in smooth_mapping.items():
            onehot_tensor[k] = v
        assert (onehot_tensor.argmax(dim=1) == torch.arange(onehot_tensor.shape[0], device=onehot_tensor.device)).all()

        self.onehot_tensor = onehot_tensor
        self.ce = nn.CrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        target = self.onehot_tensor[target[:, 0].long()].permute(0, -1, *(i + 1 for i in range(target.ndim-2))).to(device=net_output.device)

        dc_loss = self.dc(net_output, target, loss_mask=None) if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


if __name__ == '__main__':
    loss_func = DC_and_CE_smooth_loss({'smooth': 1e-5, 'do_bg': False}, {},
                                        dice_class=MemoryEfficientSoftDiceLoss)
    # print(loss_func.smooth_mapping)
    # print(loss_func(torch.randn(2, 47, 8, 8, 8), torch.randint(0, 47, (2, 1, 8, 8, 8))))
