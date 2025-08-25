import warnings

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


class SoftmaxMSELoss(_Loss):
    def __init__(
        self,
        do_softmax: bool = True,
        include_background: bool = True,
        reduction: str = "mean",
    ) -> None:
        super(SoftmaxMSELoss, self).__init__(reduction=reduction)
        self.do_softmax = do_softmax
        self.include_background = include_background

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.include_background:
            if input.shape[1] == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # If skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        assert input.requires_grad and not target.requires_grad, "make sure target has no gradient"

        if self.do_softmax:
            input = F.softmax(input, 1)
            target = F.softmax(target, 1)
        return F.mse_loss(input, target, reduction=self.reduction)
