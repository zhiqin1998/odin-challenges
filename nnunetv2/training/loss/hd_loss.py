import warnings

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from nnunetv2.training.loss.utils import one_hot
try:
    import cupy as cp
    from cucim.core.operations.morphology import distance_transform_edt
    has_cp = True
except ImportError:
    warnings.warn('Using HD loss with distance transform map has relatively high computational cost!'
                  ' Consider gpu speedup with cupy and cucim')
    import numpy as cp
    from scipy.ndimage import distance_transform_edt
    has_cp = False


class LossWrapper(nn.Module):
    def __init__(self, hd_loss, original_loss, deep_supervision=True, hd_weight=0., original_weight=1.):
        super(LossWrapper, self).__init__()
        self.hd_loss = hd_loss
        self.original_loss = original_loss
        self.hd_weight = hd_weight
        self.original_weight = original_weight
        self.deep_supervision = deep_supervision

    def forward(self, *args):
        original_loss = self.original_loss(*args)
        # only run hd loss on largest input/target (index 0) due to time complexity
        hd_loss = self.hd_loss(*next(zip(*args))) if self.deep_supervision else self.hd_loss(*args)
        return self.original_weight * original_loss + self.hd_weight * hd_loss


class HausdorffDTLoss(_Loss):
    """
    Compute channel-wise binary Hausdorff loss based on distance transform. It can support both multi-classes and
    multi-labels tasks. The data `input` (BNHW[D] where N is number of classes) is compared with ground truth `target`
    (BNHW[D]).

    Note that axis N of `input` is expected to be logits or probabilities for each class, if passing logits as input,
    must set `sigmoid=True` or `softmax=True`, or specifying `other_act`. And the same axis of `target`
    can be 1 or N (one-hot format).

    The original paper: Karimi, D. et. al. (2019) Reducing the Hausdorff Distance in Medical Image Segmentation with
    Convolutional Neural Networks, IEEE Transactions on medical imaging, 39(2), 499-513
    """

    def __init__(
        self,
        alpha: float = 2.0,
        include_background: bool = False,
        reduction: str = "mean",
        batch: bool = False,
    ) -> None:
        """
        Args:
            include_background: if False, channel index 0 (background category) is excluded from the calculation.
                if the non-background segmentations are small compared to the total image size they can get overwhelmed
                by the signal from the background so excluding it in such cases helps convergence.
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `input` (``input.shape[1]``). Defaults to False.
            sigmoid: if True, apply a sigmoid function to the prediction.
            softmax: if True, apply a softmax function to the prediction.
            other_act: callable function to execute other activation layers, Defaults to ``None``. for example:
                ``other_act = torch.tanh``.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            batch: whether to sum the intersection and union areas over the batch dimension before the dividing.
                Defaults to False, a loss value is computed independently from each item in the batch
                before any `reduction`.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        super(HausdorffDTLoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.include_background = include_background
        self.batch = batch

    @torch.no_grad()
    def distance_field(self, img: torch.Tensor) -> torch.Tensor:
        """Generate distance transform.

        Args:
            img (np.ndarray): input mask as NCHWD or NCHW.

        Returns:
            np.ndarray: Distance field.
        """
        field = torch.zeros_like(img)

        for batch_idx in range(len(img)):
            fg_mask = img[batch_idx] > 0.5

            # For cases where the mask is entirely background or entirely foreground
            # the distance transform is not well defined for all 1s,
            # which always would happen on either foreground or background, so skip
            if fg_mask.any() and not fg_mask.all():
                fg_dist: torch.Tensor = torch_distance_transform_edt(fg_mask)  # type: ignore
                bg_mask = ~fg_mask
                bg_dist: torch.Tensor = torch_distance_transform_edt(bg_mask)  # type: ignore

                field[batch_idx] = fg_dist + bg_dist

        return field

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNHW[D], where N is the number of classes.
            target: the shape should be BNHW[D] or B1HW[D], where N is the number of classes.

        Raises:
            ValueError: If the input is not 2D (NCHW) or 3D (NCHWD).
            AssertionError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].

        Example:
            >>> import torch
            >>> from monai.losses.hausdorff_loss import HausdorffDTLoss
            >>> from monai.networks.utils import one_hot
            >>> B, C, H, W = 7, 5, 3, 2
            >>> input = torch.rand(B, C, H, W)
            >>> target_idx = torch.randint(low=0, high=C - 1, size=(B, H, W)).long()
            >>> target = one_hot(target_idx[:, None, ...], num_classes=C)
            >>> self = HausdorffDTLoss(reduction='none')
            >>> loss = self(input, target)
            >>> assert np.broadcast_shapes(loss.shape, input.shape) == input.shape
        """
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError("Only 2D (NCHW) and 3D (NCHWD) supported")

        n_pred_ch = input.shape[1]

        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
        else:
            target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # If skipping background, removing first channel
                target = target[:, 1:]
                input = input[:, 1:]

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

        device = input.device
        all_f = []
        for i in range(input.shape[1]):
            ch_input = input[:, [i]]
            ch_target = target[:, [i]]
            pred_dt = self.distance_field(ch_input.detach()).float()
            target_dt = self.distance_field(ch_target.detach()).float()

            pred_error = (ch_input - ch_target) ** 2
            distance = pred_dt**self.alpha + target_dt**self.alpha

            running_f = pred_error * distance.to(device)
            reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
            if self.batch:
                # reducing spatial dimensions and batch
                reduce_axis = [0] + reduce_axis
            all_f.append(running_f.mean(dim=reduce_axis, keepdim=True))
        f = torch.cat(all_f, dim=1)
        if self.reduction == "mean":
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == "sum":
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == "none":
            # If we are not computing voxelwise loss components at least make sure a none reduction maintains a
            # broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(ch_input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f


def torch_distance_transform_edt(
    img,
    sampling: None | float | list[float] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances = None,
    indices = None,
    *,
    block_params: tuple[int, int, int] | None = None,
    float64_distances: bool = False):
    """
    Euclidean distance transform, either GPU based with CuPy / cuCIM or CPU based with scipy.
    To use the GPU implementation, make sure cuCIM is available and that the data is a `torch.tensor` on a GPU device.

    Note that the results of the libraries can differ, so stick to one if possible.
    For details, check out the `SciPy`_ and `cuCIM`_ documentation.

    .. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html
    .. _cuCIM: https://docs.rapids.ai/api/cucim/nightly/api/#cucim.core.operations.morphology.distance_transform_edt

    Args:
        img: Input image on which the distance transform shall be run.
            Has to be a channel first array, must have shape: (num_channels, H, W [,D]).
            Can be of any type but will be converted into binary: 1 wherever image equates to True, 0 elsewhere.
            Input gets passed channel-wise to the distance-transform, thus results from this function will differ
            from directly calling ``distance_transform_edt()`` in CuPy or SciPy.
        sampling: Spacing of elements along each dimension. If a sequence, must be of length equal to the input rank -1;
            if a single number, this is used for all axes. If not specified, a grid spacing of unity is implied.
        return_distances: Whether to calculate the distance transform.
        return_indices: Whether to calculate the feature transform.
        distances: An output array to store the calculated distance transform, instead of returning it.
            `return_distances` must be True.
        indices: An output array to store the calculated feature transform, instead of returning it. `return_indicies` must be True.
        block_params: This parameter is specific to cuCIM and does not exist in SciPy. For details, look into `cuCIM`_.
        float64_distances: This parameter is specific to cuCIM and does not exist in SciPy.
            If True, use double precision in the distance computation (to match SciPy behavior).
            Otherwise, single precision will be used for efficiency.

    Returns:
        distances: The calculated distance transform. Returned only when `return_distances` is True and `distances` is not supplied.
            It will have the same shape and type as image. For cuCIM: Will have dtype torch.float64 if float64_distances is True,
            otherwise it will have dtype torch.float32. For SciPy: Will have dtype np.float64.
        indices: The calculated feature transform. It has an image-shaped array for each dimension of the image.
            The type will be equal to the type of the image.
            Returned only when `return_indices` is True and `indices` is not supplied. dtype np.float64.

    """

    if not return_distances and not return_indices:
        raise RuntimeError("Neither return_distances nor return_indices True")

    if not (img.ndim >= 3 and img.ndim <= 4):
        raise RuntimeError("Wrong input dimensionality. Use (num_channels, H, W [,D])")

    distances_original, indices_original = distances, indices
    distances, indices = None, None
    if has_cp:
        distances_, indices_ = None, None
        if return_distances:
            dtype = torch.float64 if float64_distances else torch.float32
            if distances is None:
                distances = torch.zeros_like(img, memory_format=torch.contiguous_format, dtype=dtype)  # type: ignore
            else:
                if not isinstance(distances, torch.Tensor) and distances.device != img.device:
                    raise TypeError("distances must be a torch.Tensor on the same device as img")
                if not distances.dtype == dtype:
                    raise TypeError("distances must be a torch.Tensor of dtype float32 or float64")
            distances_ = convert_to_cupy(distances)
        if return_indices:
            dtype = torch.int32
            if indices is None:
                indices = torch.zeros((img.dim(),) + img.shape, dtype=dtype)  # type: ignore
            else:
                if not isinstance(indices, torch.Tensor) and indices.device != img.device:
                    raise TypeError("indices must be a torch.Tensor on the same device as img")
                if not indices.dtype == dtype:
                    raise TypeError("indices must be a torch.Tensor of dtype int32")
            indices_ = convert_to_cupy(indices)
        img_ = convert_to_cupy(img)
        for channel_idx in range(img_.shape[0]):
            distance_transform_edt(
                img_[channel_idx],
                sampling=sampling,
                return_distances=return_distances,
                return_indices=return_indices,
                distances=distances_[channel_idx] if distances_ is not None else None,
                indices=indices_[channel_idx] if indices_ is not None else None,
                block_params=block_params,
                float64_distances=float64_distances,
            )
        torch.cuda.synchronize()
    else:
        img_ = convert_to_cupy(img)
        if return_distances:
            if distances is None:
                distances = cp.zeros_like(img_, dtype=cp.float64)
            else:
                if not isinstance(distances, cp.ndarray):
                    raise TypeError("distances must be a numpy.ndarray")
                if not distances.dtype == cp.float64:
                    raise TypeError("distances must be a numpy.ndarray of dtype float64")
        if return_indices:
            if indices is None:
                indices = cp.zeros((img_.ndim,) + img_.shape, dtype=cp.int32)
            else:
                if not isinstance(indices, cp.ndarray):
                    raise TypeError("indices must be a numpy.ndarray")
                if not indices.dtype == cp.int32:
                    raise TypeError("indices must be a numpy.ndarray of dtype int32")

        for channel_idx in range(img_.shape[0]):
            distance_transform_edt(
                img_[channel_idx],
                sampling=sampling,
                return_distances=return_distances,
                return_indices=return_indices,
                distances=distances[channel_idx] if distances is not None else None,
                indices=indices[channel_idx] if indices is not None else None,
            )

    r_vals = []
    if return_distances and distances_original is None:
        r_vals.append(distances_ if has_cp else distances)
    if return_indices and indices_original is None:
        r_vals.append(indices)
    if not r_vals:
        return None
    device = img.device
    return torch.as_tensor(r_vals[0] if len(r_vals) == 1 else r_vals, device=device)
    # return convert_data_type(r_vals[0] if len(r_vals) == 1 else r_vals, output_type=type(img), device=device)[0]


def convert_to_cupy(data, dtype=None):
    if has_cp:
        if data.dtype == torch.bool:
            data = data.detach().to(torch.uint8)
            if dtype is None:
                dtype = bool  # type: ignore
    else:
        data = data.cpu()
    data = cp.asarray(data, dtype)
    if data.ndim > 0:
        data = cp.ascontiguousarray(data)
    return data


if __name__ == "__main__":
    import torch
    import time
    a = torch.randn(20, 256, 256).cuda()
    b = torch.randint(0, a.shape[0], (1, *a.shape[1:])).cuda()
    x = torch_distance_transform_edt(a)
    print(x, x.shape)
    st = time.time()
    x = HausdorffDTLoss()(a[None], b[None])
    print(x)
    print(time.time() - st)
