import torch
from torch import nn

from mamba_ssm import Mamba, Mamba2
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
# pip install --no-build-isolation "git+https://github.com/state-spaces/mamba@v2.2.5"
from torch.amp import autocast
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.nets.point_encoder import PointEncoder, PositionEmbeddingRandom
from nnunetv2.nets.attention import CrossAttentionBlock, Attention


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, use_fast_path=True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            use_fast_path=use_fast_path
        )

    # @autocast('cuda', enabled=False)
    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class MambaLayer2(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2, headdim=80, use_mem_eff_path=True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=headdim, # make sure dim*expand/headdim % 8 = 0
            use_mem_eff_path=use_mem_eff_path
        )

    # @autocast('cuda', enabled=False)
    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class MambaLayer2PE(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2, headdim=80, use_mem_eff_path=True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=headdim, # make sure dim*expand/headdim % 8 = 0
            use_mem_eff_path=use_mem_eff_path
        )
        self.position_encoder = PositionEmbeddingRandom(dim // 2)

    # @autocast('cuda', enabled=False)
    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        B, C = x.shape[:2]
        # assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        # img_pe = self.position_encoder(img_dims).unsqueeze(0)  # 1 x C x H x W x D
        # img_pe = torch.repeat_interleave(self.position_encoder(img_dims).unsqueeze(0), B, dim=0).reshape(B, C, n_tokens).transpose(-1, -2)  # B x n_tokens x C
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) + torch.repeat_interleave(self.position_encoder(img_dims).unsqueeze(0), B, dim=0).reshape(B, C, n_tokens).transpose(-1, -2)  # B x n_tokens x C
        x_flat = self.norm(x_flat)
        x_flat = self.mamba(x_flat)
        # out = x_flat.transpose(-1, -2).reshape(B, C, *img_dims)

        return x_flat.transpose(-1, -2).reshape(B, C, *img_dims)


class UMambaBot(ResidualEncoderUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer(dim=self.encoder.output_channels[-1])

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


class UMambaBot2(ResidualEncoderUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer2(dim=self.encoder.output_channels[-1])

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)

class UMambaBot2PE(ResidualEncoderUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer2PE(dim=self.encoder.output_channels[-1])

    def forward(self, x):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1])
        return self.decoder(skips)


class MambaLayer2_Click(nn.Module):
    def __init__(self, dim, patch_size, num_point_embeddings=2, d_state=32, d_conv=4, expand=2, headdim=80, use_mem_eff_path=True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=headdim, # make sure dim*expand/headdim % 8 = 0
            use_mem_eff_path=use_mem_eff_path
        )
        self.point_encoder = PointEncoder(dim, patch_size, num_point_embeddings)

    # points have variable size, cannot compile properly
    # half precision causes nan
    @autocast('cuda', enabled=False)
    @torch.compiler.disable
    def forward(self, x, points):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            points = (points[0].type(torch.float32), points[1].type(torch.float32))
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        point_emb = self.point_encoder(points) # BxNxC
        x_flat = torch.concat((x_flat, point_emb), dim=1) # concat and let mamba figure out the image feat + point emb
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)[:, :n_tokens] # discard point embeddings
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class UMambaBot2_Click(ResidualEncoderUNet):
    def __init__(self, *args, patch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer2_Click(dim=self.encoder.output_channels[-1], patch_size=patch_size)

    def forward(self, x, points=None):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1], points)
        return self.decoder(skips)


class MambaLayer2_CrossAttn_Click(nn.Module):
    def __init__(self, dim, patch_size, num_point_embeddings=2, d_state=32, d_conv=4, expand=2, headdim=80,
                 use_mem_eff_path=True, attn_num_heads=4):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=headdim, # make sure dim*expand/headdim % 8 = 0
            use_mem_eff_path=use_mem_eff_path
        )
        self.point_encoder = PointEncoder(dim, patch_size, num_point_embeddings)
        self.cross_attention = CrossAttentionBlock(dim, attn_num_heads, skip_first_layer_pe=True)

    # points have variable size, cannot compile properly
    # half precision causes nan
    @autocast('cuda', enabled=False)
    @torch.compiler.disable
    def forward(self, x, points):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            points = (points[0].type(torch.float32), points[1].type(torch.float32))
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # cross attend point and image with position encoding
        point_emb = self.point_encoder(points) # BxNxC
        img_pe = self.point_encoder.pe_layer(img_dims).unsqueeze(0) # 1 x C x H x W x D
        img_pe = torch.repeat_interleave(img_pe, B, dim=0).reshape(B, C, n_tokens).transpose(-1, -2) # B x n_tokens x C
        point_emb, x_mamba = self.cross_attention(
            queries=point_emb,
            keys=x_mamba,
            query_pe=point_emb,
            key_pe=img_pe,
        ) # already layer norm within cross attention block
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class UMambaBot2_CrossAttn_Click(ResidualEncoderUNet):
    def __init__(self, *args, patch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer2_CrossAttn_Click(dim=self.encoder.output_channels[-1], patch_size=patch_size)

    def forward(self, x, points=None):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1], points)
        return self.decoder(skips)

class MambaLayer2_CrossAttn_Click2(nn.Module):
    def __init__(self, dim, patch_size, num_point_embeddings=2, d_state=32, d_conv=4, expand=2, headdim=80,
                 use_mem_eff_path=True, attn_num_heads=4, attn_depth=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba2Simple(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            headdim=headdim, # make sure dim*expand/headdim % 8 = 0
            use_mem_eff_path=use_mem_eff_path
        )
        self.point_encoder = PointEncoder(dim, patch_size, num_point_embeddings)
        self.cross_attention = nn.ModuleList()
        for i in range(attn_depth):
            self.cross_attention.append(
                CrossAttentionBlock(
                    embedding_dim=dim,
                    num_heads=attn_num_heads,
                    mlp_dim=1024,
                    skip_first_layer_pe=(i == 0),
                )
            )

    # points have variable size, cannot compile properly
    # half precision causes nan
    @autocast('cuda', enabled=False)
    @torch.compiler.disable
    def forward(self, x, points):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            points = (points[0].type(torch.float32), points[1].type(torch.float32))
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        # cross attend point and image with position encoding
        point_emb = self.point_encoder(points) # BxNxC
        img_pe = self.point_encoder.pe_layer(img_dims).unsqueeze(0) # 1 x C x H x W x D
        img_pe = torch.repeat_interleave(img_pe, B, dim=0).reshape(B, C, n_tokens).transpose(-1, -2) # B x n_tokens x C

        queries = point_emb
        keys = x_mamba
        for layer in self.cross_attention:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_emb,
                key_pe=img_pe,
            )
        out = keys.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class UMambaBot2_CrossAttn_Click2(ResidualEncoderUNet):
    def __init__(self, *args, patch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba_layer = MambaLayer2_CrossAttn_Click2(dim=self.encoder.output_channels[-1], patch_size=patch_size)

    def forward(self, x, points=None):
        skips = self.encoder(x)
        skips[-1] = self.mamba_layer(skips[-1], points)
        return self.decoder(skips)

if __name__ == "__main__":
    arch_kwargs = {
        "n_stages": 7,
        "features_per_stage": [
            32,
            64,
            128,
            256,
            320,
            320,
            320
        ],
        "conv_op": torch.nn.modules.conv.Conv3d,
        "kernel_sizes": [
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ],
            [
                3,
                3,
                3
            ]
        ],
        "strides": [
            [
                1,
                1,
                1
            ],
            [
                2,
                2,
                2
            ],
            [
                2,
                2,
                2
            ],
            [
                2,
                2,
                2
            ],
            [
                2,
                2,
                2
            ],
            [
                2,
                2,
                2
            ],
            [
                1,
                2,
                2
            ]
        ],
        "n_blocks_per_stage": [
            1,
            3,
            4,
            6,
            6,
            6,
            6
        ],
        "n_conv_per_stage_decoder": [
            1,
            1,
            1,
            1,
            1,
            1
        ],
        "conv_bias": True,
        "norm_op": torch.nn.modules.instancenorm.InstanceNorm3d,
        "norm_op_kwargs": {
            "eps": 1e-05,
            "affine": True
        },
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": torch.nn.LeakyReLU,
        "nonlin_kwargs": {
            "inplace": True
        },
        "patch_size": [
            128,
            256,
            256
        ],
        "input_channels": 1,
        "num_classes": 3,
        "deep_supervision": True,
    }
    with torch.no_grad():
        model = UMambaBot2_CrossAttn_Click(**arch_kwargs).cuda()
        input = torch.randn((1, 1, 128, 256, 256)).cuda()
        points = torch.as_tensor([[30, 60, 120], [56, 50, 179], [90, 160, 2]]).reshape(1, -1, 3).cuda()
        labels = torch.as_tensor([[0, 1, 1]]).reshape(1, -1).cuda()
        out = model(input, (points, labels))
