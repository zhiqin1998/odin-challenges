import warnings

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
import numpy as np
from nnunetv2.training.loss.utils import one_hot


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()


class FocalLoss(_Loss):
    """
    Focal loss adapted from monai 1.4.0
    """

    def __init__(
        self,
        include_background: bool = True,
        gamma: float = 2.0,
        alpha: float | None = None,
        use_softmax = True,
        weight: float | int | torch.Tensor | None = None,
        reduction:  str = "mean",
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the loss calculation.
                If False, `alpha` is invalid when using softmax.
            to_onehot_y: whether to convert the label `y` into the one-hot format. Defaults to False.
            gamma: value of the exponent gamma in the definition of the Focal loss. Defaults to 2.
            alpha: value of the alpha in the definition of the alpha-balanced Focal loss.
                The value should be in [0, 1]. Defaults to None.
            weight: weights to apply to the voxels of each class. If None no weights are applied.
                The input can be a single value (same weight for all classes), a sequence of values (the length
                of the sequence should be the same as the number of classes. If not ``include_background``,
                the number of classes should not include the background category class 0).
                The value/values should be no less than 0. Defaults to None.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.

        """
        super().__init__(reduction=reduction)
        self.include_background = include_background
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        weight = torch.as_tensor(weight) if weight is not None else None
        self.register_buffer("class_weight", weight)
        self.class_weight: None | torch.Tensor
        self.use_softmax = use_softmax

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                The input should be the original logits since it will be transformed by
                a sigmoid/softmax in the forward function.
            target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.

        Raises:
            ValueError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
            ValueError: When ``self.weight`` is a sequence and the length is not equal to the
                number of classes.
            ValueError: When ``self.weight`` is/contains a value that is less than 0.

        """
        n_pred_ch = input.shape[1]

        if n_pred_ch == 1:
            warnings.warn("single channel prediction, one_hot ignored.")
        else:
            target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise ValueError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        input = input.float()
        target = target.float()
        if self.use_softmax:
            if not self.include_background and self.alpha is not None:
                self.alpha = None
                warnings.warn("`include_background=False`, `alpha` ignored when using softmax.")
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

        num_of_classes = target.shape[1]
        if self.class_weight is not None and num_of_classes != 1:
            # make sure the lengths of weights are equal to the number of classes
            if self.class_weight.ndim == 0:
                self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
            else:
                if self.class_weight.shape[0] != num_of_classes:
                    raise ValueError(
                        """the length of the `weight` sequence should be the same as the number of classes.
                        If `include_background=False`, the weight should not include
                        the background category class 0."""
                    )
            if self.class_weight.min() < 0:
                raise ValueError("the value/values of the `weight` should be no less than 0.")
            # apply class_weight to loss
            self.class_weight = self.class_weight.to(loss)
            broadcast_dims = [-1] + [1] * len(target.shape[2:])
            self.class_weight = self.class_weight.view(broadcast_dims)
            loss = self.class_weight * loss

        if self.reduction == "sum":
            # Previously there was a mean over the last dimension, which did not
            # return a compatible BCE loss. To maintain backwards compatible
            # behavior we have a flag that performs this extra step, disable or
            # parameterize if necessary. (Or justify why the mean should be there)
            average_spatial_dims = True
            if average_spatial_dims:
                loss = loss.mean(dim=list(range(2, len(target.shape))))
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return loss


def softmax_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls = input.log_softmax(1)
    loss: torch.Tensor = -(1 - input_ls.exp()).pow(gamma) * input_ls * target

    if alpha is not None:
        # (1-alpha) for the background class and alpha for the other classes
        alpha_fac = torch.tensor([1 - alpha] + [alpha] * (target.shape[1] - 1)).to(loss)
        broadcast_dims = [-1] + [1] * len(target.shape[2:])
        alpha_fac = alpha_fac.view(broadcast_dims)
        loss = alpha_fac * loss

    return loss


def sigmoid_focal_loss(
    input: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha = None
) -> torch.Tensor:
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    # computing binary cross entropy with logits
    # equivalent to F.binary_cross_entropy_with_logits(input, target, reduction='none')
    # see also https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Loss.cpp#L363
    loss: torch.Tensor = input - input * target - F.logsigmoid(input)

    # sigmoid(-i) if t==1; sigmoid(i) if t==0 <=>
    # 1-sigmoid(i) if t==1; sigmoid(i) if t==0 <=>
    # 1-p if t==1; p if t==0 <=>
    # pfac, that is, the term (1 - pt)
    invprobs = F.logsigmoid(-input * (target * 2 - 1))  # reduced chance of overflow
    # (pfac.log() * gamma).exp() <=>
    # pfac.log().exp() ^ gamma <=>
    # pfac ^ gamma
    loss = (invprobs * gamma).exp() * loss

    if alpha is not None:
        # alpha if t==1; (1-alpha) if t==0
        alpha_factor = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss

    return loss