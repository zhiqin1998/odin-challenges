# this file is implemented based on SAM
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, Any

import torch
from torch import nn


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 3 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        x_embed = grid.cumsum(dim=0) - 0.5
        y_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        x_embed = x_embed / h
        y_embed = y_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x H x W x D

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[0]
        coords[:, :, 1] = coords[:, :, 1] / image_size[1]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class PointEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_image_size: Tuple[int, int, int],
        num_point_embeddings: int = 2,
    ) -> None:
        """
        Encodes point for input following SAM-style prompt encoder

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W, D).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = num_point_embeddings  # left/right point
        point_embeddings = [
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad: # include at least one not a point
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        point_embedding = torch.where(
            (labels == -1).unsqueeze(-1),
            torch.zeros_like(point_embedding) + self.not_a_point_embed.weight,
            point_embedding,
        )
        assert labels.max() < self.num_point_embeddings, f'encountered invalid labels {labels}'

        for i in range(self.num_point_embeddings):
            point_embedding = torch.where(
                (labels == i).unsqueeze(-1),
                point_embedding + self.point_embeddings[i].weight,
                point_embedding,
            )
        # point_embedding = torch.where(
        #     (labels == 1).unsqueeze(-1),
        #     point_embedding + self.point_embeddings[1].weight,
        #     point_embedding,
        # )
        return point_embedding

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Embeds point prompts.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.

        Returns:
          torch.Tensor: sparse embeddings for the points, with shape
            BxNx(embed_dim), where N is determined by the number of input points
        """
        bs = points[0].shape[0] if points is not None else 1
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=True)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        return sparse_embeddings
