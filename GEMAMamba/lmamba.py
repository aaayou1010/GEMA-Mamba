
import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

from .shift_cuda import BasicLayer_mlp, MyNorm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
except Exception:
    selective_scan_fn_v1 = None


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(
    B=1,
    L=256,
    D=768,
    N=16,
    with_D=True,
    with_Z=False,
    with_Group=True,
    with_complex=False,
):
    import numpy as np

    def einsum_flops(input_shapes, equation):
        samples = [np.zeros(shape) for shape in input_shapes]
        plan = np.einsum_path(equation, *samples, optimize="optimal")[1]
        for line in plan.splitlines():
            if "optimized flop" in line.lower():
                return float(np.floor(float(line.split(":")[-1]) / 2))
        return 0.0

    assert not with_complex

    total = einsum_flops([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        total += einsum_flops([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        total += einsum_flops([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    loop_cost = B * D * N
    if with_Group:
        loop_cost += einsum_flops([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        loop_cost += einsum_flops([[B, D, N], [B, N]], "bdn,bn->bd")

    total += L * loop_cost
    if with_D:
        total += B * D * L
    if with_Z:
        total += B * D * L
    return total


def _checkpoint_seq(blocks, x, use_checkpoint):
    for block in blocks:
        x = checkpoint.checkpoint(block, x) if use_checkpoint else block(x)
    return x


def _resize_even_grid(x):
    b, h, w, c = x.shape
    target_h = h // 2
    target_w = w // 2
    tiles = [
        x[:, 0::2, 0::2, :],
        x[:, 1::2, 0::2, :],
        x[:, 0::2, 1::2, :],
        x[:, 1::2, 1::2, :],
    ]
    if (h % 2) or (w % 2):
        print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
        tiles = [tile[:, :target_h, :target_w, :] for tile in tiles]
    return b, target_h, target_w, c, tiles


def _build_directional_sequences(x):
    b, _, h, w = x.shape
    length = h * w
    horizontal = x.reshape(b, -1, length)
    vertical = x.transpose(2, 3).contiguous().reshape(b, -1, length)
    forward = torch.stack([horizontal, vertical], dim=1)
    backward = torch.flip(forward, dims=[-1])
    return torch.cat([forward, backward], dim=1), length, h, w


def _split_scan_outputs(scan_out, width, height):
    inverse = torch.flip(scan_out[:, 2:4], dims=[-1]).reshape(scan_out.size(0), 2, -1, width * height)
    vertical = scan_out[:, 1].reshape(scan_out.size(0), -1, width, height).transpose(2, 3).contiguous()
    inverse_vertical = inverse[:, 1].reshape(scan_out.size(0), -1, width, height).transpose(2, 3).contiguous()
    return scan_out[:, 0], inverse[:, 0], vertical.reshape(scan_out.size(0), -1, width * height), inverse_vertical.reshape(scan_out.size(0), -1, width * height)


def _mark_no_weight_decay(parameter):
    parameter._no_weight_decay = True
    return parameter


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        patch = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch, stride=patch)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x).permute(0, 2, 3, 1))


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)

    def forward(self, x):
        b, target_h, target_w, c, tiles = _resize_even_grid(x)
        merged = torch.cat(tiles, dim=-1).reshape(b, target_h, target_w, c * 4)
        return self.reduction(self.norm(merged))


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        expanded_dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(expanded_dim, expanded_dim * dim_scale, bias=False)
        self.norm = norm_layer(expanded_dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=x.shape[-1] // (self.dim_scale * self.dim_scale),
        )
        return self.norm(x)


class FinalPatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.dim_scale,
            p2=self.dim_scale,
            c=x.shape[-1] // (self.dim_scale * self.dim_scale),
        )
        return self.norm(x)


Final_PatchExpand2D = FinalPatchExpand2D


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_impl = selective_scan_fn

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.depthwise = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.activation = nn.SiLU()

        x_proj_layers = [
            nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False, **factory_kwargs)
            for _ in range(4)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in x_proj_layers], dim=0))

        dt_layers = [
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(4)
        ]
        self.dt_proj_weight = nn.Parameter(torch.stack([layer.weight for layer in dt_layers], dim=0))
        self.dt_proj_bias = nn.Parameter(torch.stack([layer.bias for layer in dt_layers], dim=0))

        self.A_logs = self.A_log_init(d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        layer = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(layer.weight, std)
        elif dt_init == "random":
            nn.init.uniform_(layer.weight, -std, std)
        else:
            raise NotImplementedError(f"Unsupported dt init mode: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            layer.bias.copy_(inv_dt)
        layer.bias._no_reinit = True
        return layer

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        base = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner)
        logs = torch.log(base).contiguous()
        if copies > 1:
            logs = repeat(logs, "d n -> r d n", r=copies)
            if merge:
                logs = logs.flatten(0, 1)
        return _mark_no_weight_decay(nn.Parameter(logs))

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        values = torch.ones(d_inner, device=device)
        if copies > 1:
            values = repeat(values, "n -> r n", r=copies)
            if merge:
                values = values.flatten(0, 1)
        return _mark_no_weight_decay(nn.Parameter(values))

    def _resolve_scan_impl(self):
        if self.scan_impl is not None:
            return self.scan_impl, True
        if selective_scan_fn_v1 is not None:
            return selective_scan_fn_v1, False
        raise ImportError("selective_scan implementation is unavailable. Please install mamba_ssm or selective_scan.")

    def _project_scan_params(self, xs):
        b, k, _, length = xs.shape
        projected = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.reshape(b, k, -1, length), self.dt_proj_weight)
        return (
            xs.float().reshape(b, -1, length),
            dts.contiguous().float().reshape(b, -1, length),
            Bs.float().reshape(b, k, -1, length),
            Cs.float().reshape(b, k, -1, length),
        )

    def _run_scan(self, xs, dts, Bs, Cs, use_primary_impl):
        scan_impl, _ = self._resolve_scan_impl()
        kwargs = {
            "delta_bias": self.dt_proj_bias.float().reshape(-1),
            "delta_softplus": True,
        }
        if use_primary_impl:
            kwargs["z"] = None
            kwargs["return_last_state"] = False
        return scan_impl(
            xs,
            dts,
            -torch.exp(self.A_logs.float()).reshape(-1, self.d_state),
            Bs,
            Cs,
            self.Ds.float().reshape(-1),
            **kwargs,
        )

    def _scan(self, x):
        sequences, length, height, width = _build_directional_sequences(x)
        xs, dts, Bs, Cs = self._project_scan_params(sequences)
        _, use_primary = self._resolve_scan_impl()
        scan_out = self._run_scan(xs, dts, Bs, Cs, use_primary).reshape(x.size(0), 4, -1, length)
        y1, y2, y3, y4 = _split_scan_outputs(scan_out, width, height)
        return y1 + y2 + y3 + y4

    def forward(self, x, **kwargs):
        b, h, w, _ = x.shape
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.activation(self.depthwise(value.permute(0, 3, 1, 2).contiguous()))
        fused = self._scan(value).transpose(1, 2).contiguous().reshape(b, h, w, -1)
        fused = self.out_norm(fused) * F.silu(gate)
        return self.dropout(self.out_proj(fused))


class GatedMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        inner = hidden_dim or int(dim * 1.5)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.dropout(x)
        return self.dropout(self.out_proj(x))


class SSGMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim=0,
        drop_path=0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate=0.0,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.attn_norm = norm_layer(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ssm = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.ffn = GatedMLP(hidden_dim, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ssm(self.attn_norm(x)))
        x = x + self.drop_path(self.ffn(self.ffn_norm(x)))
        return x


class LearnableSkipRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        routed_dim = dim // 4
        self.norm = nn.LayerNorm(dim)
        self.skip_value = nn.Linear(dim, routed_dim, bias=False)
        self.skip_gate = nn.Linear(dim, routed_dim, bias=True)
        self.aux_value = nn.Linear(dim, routed_dim, bias=False)
        self.aux_gate = nn.Linear(dim, routed_dim, bias=True)

    def forward(self, x):
        x = self.norm(x)
        skip = self.skip_value(x) * torch.sigmoid(self.skip_gate(x))
        aux = self.aux_value(x) * torch.sigmoid(self.aux_gate(x))
        return skip, aux


class AFSM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.context = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.enc_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.dec_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, enc, dec):
        base = self.context(torch.cat([enc, dec], dim=1))
        gated = enc * self.enc_gate(base) + dec * self.dec_gate(base)
        return self.mix(torch.cat([base, gated], dim=1)) + dec


class LocalContextGatedRefinementModule(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.context = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x_nchw = x.permute(0, 3, 1, 2)
        local = self.local(x_nchw)
        context = self.context(x_nchw)
        fused = self.fuse(torch.cat([local, context], dim=1))
        fused = fused * self.channel_gate(fused)
        avg_map = fused.mean(dim=1, keepdim=True)
        max_map, _ = fused.max(dim=1, keepdim=True)
        fused = fused * self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        return (fused + local).permute(0, 2, 3, 1)


def _reset_out_proj_weights(module):
    for name, parameter in module.named_parameters():
        if name == "out_proj.weight":
            cloned = parameter.clone().detach_()
            nn.init.kaiming_uniform_(cloned, a=math.sqrt(5))


class VSSLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SSGMBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )
        self.local_context_refinement = LocalContextGatedRefinementModule(dim)
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
        self.apply(_reset_out_proj_weights)

    def forward(self, x):
        x = x + self.local_context_refinement(x)
        x = _checkpoint_seq(self.blocks, x, self.use_checkpoint)
        return self.downsample(x) if self.downsample is not None else x


class VSSLayerUp(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample is not None else None
        self.blocks = nn.ModuleList(
            [
                SSGMBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                )
                for i in range(depth)
            ]
        )
        self.apply(_reset_out_proj_weights)

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        return _checkpoint_seq(self.blocks, x, self.use_checkpoint)


VSSLayer_up = VSSLayerUp


class ACFA(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        split_channels = max(channels // 4, 1)
        hidden = max(channels // reduction, 4)
        self.channel_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.channel_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial = nn.Conv2d(channels, 1, kernel_size=7, padding=3, bias=True)
        self.pre_proj = nn.Conv2d(channels, split_channels * 4, kernel_size=1, bias=False)
        self.hw = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, 1, bias=False),
            nn.Conv2d(split_channels, split_channels, 3, padding=1, groups=split_channels, bias=False),
        )
        self.h_branch = nn.Conv1d(split_channels, split_channels, 3, padding=1, groups=split_channels, bias=False)
        self.w_branch = nn.Conv1d(split_channels, split_channels, 3, padding=1, groups=split_channels, bias=False)
        self.tail = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, 3, padding=1, groups=split_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(split_channels, split_channels, 1, bias=False),
        )
        self.norm = nn.LayerNorm(split_channels * 4)
        self.out_proj = nn.Conv2d(split_channels * 4, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, _, h, w = x.shape
        gated = x * torch.sigmoid(self.channel_avg(x) + self.channel_max(x)) * torch.sigmoid(self.spatial(x))
        parts = torch.chunk(self.pre_proj(gated), 4, dim=1)
        h_feat = F.gelu(self.h_branch(parts[1].mean(dim=3))).unsqueeze(-1).expand(-1, -1, h, w)
        w_feat = F.gelu(self.w_branch(parts[2].mean(dim=2))).unsqueeze(2).expand(-1, -1, h, w)
        fused = torch.cat([self.hw(parts[0]), h_feat, w_feat, self.tail(parts[3])], dim=1)
        fused = self.norm(fused.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(b, -1, h, w)
        return self.out_proj(fused) + x


class TFFALite(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 4)
        self.spatial = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, bias=False), nn.GELU())
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, bias=False),
        )
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim * 2, kernel_size=1, bias=True),
        )
        self.mix_logits = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, 3, kernel_size=1, bias=True),
        )
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.acfa = ACFA(dim, reduction=reduction)
        self.merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )

    def _frequency_branch(self, x):
        spectrum = torch.fft.fft2(x, norm="ortho")
        pooled = x.abs().mean(dim=(2, 3), keepdim=True)
        low_scale, high_scale = torch.sigmoid(self.freq_gate(pooled)).chunk(2, dim=1)
        h, w = x.shape[-2:]
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        radius = torch.sqrt(xx * xx + yy * yy)
        radius = radius / radius.amax().clamp_min(1e-6)
        low = torch.fft.ifft2(spectrum * (1.0 - radius), norm="ortho").real * low_scale
        high = torch.fft.ifft2(spectrum * radius, norm="ortho").real * high_scale
        return low + high

    def forward(self, x):
        branches = [self.spatial(x), self.local(x), self._frequency_branch(x)]
        weights = F.softmax(self.mix_logits(torch.cat(branches, dim=1)), dim=1)
        fused = sum(weights[:, idx : idx + 1] * branch for idx, branch in enumerate(branches))
        fused = self.out_proj(self.norm(fused))
        return self.merge(torch.cat([fused, self.acfa(x)], dim=1)) + x


class MultiScalePerception(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class ChannelGatingSaliency(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        return x * self.channel_gate(x) * self.spatial_gate(torch.cat([avg_map, max_map], dim=1))


class SMMM(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        self.multi_scale = MultiScalePerception(channels)
        self.saliency = ChannelGatingSaliency(channels)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, enc, dec):
        fused = self.saliency(self.multi_scale(enc)) + self.saliency(self.multi_scale(dec))
        fused = F.relu(self.fuse(fused), inplace=True)
        return self.out_proj(self.norm(fused))


class MSC(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, proj_drop=0.0, **kwargs):
        super().__init__()
        half = dim // 2
        cnn_dim = half * 2
        pool_dim = half * 2
        self.cnn_in = half
        self.conv_branch = nn.Sequential(
            nn.Conv2d(half, cnn_dim, kernel_size=1, bias=False),
            nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=cnn_dim, bias=False),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.pool_proj = nn.Sequential(nn.Conv2d(half, pool_dim, kernel_size=1), nn.GELU())
        self.mix = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=False, groups=dim)
        self.out_proj = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.dropout = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        conv_part = self.conv_branch(x[:, : self.cnn_in])
        pool_part = self.pool_proj(self.pool(x[:, self.cnn_in :]))
        x = torch.cat([conv_part, pool_part], dim=1)
        x = self.out_proj(x + self.mix(x))
        return self.dropout(x).permute(0, 2, 3, 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, hidden, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, in_planes, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        weight = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return weight.permute(0, 2, 3, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1))).permute(0, 2, 3, 1)


class VSSM(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2],
        dims=[96, 192, 384, 768],
        dims_decoder=[768, 384, 192, 96],
        d_state=16,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** layer_idx) for layer_idx in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.ape = False

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_drop_path = [item.item() for item in torch.linspace(0, drop_path_rate, sum(depths))]
        decoder_drop_path = [item.item() for item in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        state_dim = math.ceil(dims[0] / 6) if d_state is None else d_state

        self.layers = nn.ModuleList(
            [
                VSSLayer(
                    dim=dims[layer_idx],
                    depth=1,
                    d_state=state_dim,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=encoder_drop_path[sum(depths[:layer_idx]) : sum(depths[: layer_idx + 1])],
                    norm_layer=norm_layer,
                    downsample=PatchMerging2D if layer_idx < self.num_layers - 1 else None,
                    use_checkpoint=use_checkpoint,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.skip_router = nn.ModuleList([LearnableSkipRouter(dim) for dim in dims])
        self.asc = nn.ModuleList(
            [
                BasicLayer_mlp(
                    dim=dims[layer_idx] // 4,
                    input_resolution=(0, 0),
                    depth=depths[layer_idx],
                    shift_size=1,
                    mlp_ratio=1,
                    drop=drop_rate,
                    drop_path=encoder_drop_path[sum(depths[:layer_idx]) : sum(depths[: layer_idx + 1])],
                    norm_layer=MyNorm,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.layers_up = nn.ModuleList(
            [
                VSSLayerUp(
                    dim=dims_decoder[layer_idx],
                    depth=depths_decoder[layer_idx],
                    d_state=state_dim,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=decoder_drop_path[
                        sum(depths_decoder[:layer_idx]) : sum(depths_decoder[: layer_idx + 1])
                    ],
                    norm_layer=norm_layer,
                    upsample=PatchExpand2D if layer_idx != 0 else None,
                    use_checkpoint=use_checkpoint,
                )
                for layer_idx in range(self.num_layers)
            ]
        )

        self.skip_fuse = nn.ModuleList([AFSM(dim, reduction=8) for dim in dims_decoder[:3]])
        self.final_up = FinalPatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def _embed(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self._embed(x)
        skip_tokens = []
        aux_tokens = []
        for stage, aux_stage, router in zip(self.layers, self.asc, self.skip_router):
            skip, aux = router(x)
            skip_tokens.append(skip)
            aux_tokens.append(aux_stage(aux))
            x = stage(x)
        return x, skip_tokens, aux_tokens

    def forward_features_up(self, x, skip_tokens, aux_tokens):
        for idx, stage in enumerate(self.layers_up):
            if idx == 0:
                x = stage(x)
                continue
            enc = torch.cat([skip_tokens[-idx], aux_tokens[-idx]], dim=3).permute(0, 3, 1, 2)
            dec = x.permute(0, 3, 1, 2)
            x = stage(self.skip_fuse[idx - 1](enc, dec).permute(0, 2, 3, 1))
        return x

    def forward_final(self, x):
        x = self.final_up(x).permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward_backbone(self, x):
        x = self._embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_tokens, aux_tokens = self.forward_features(x)
        x = self.forward_features_up(x, skip_tokens, aux_tokens)
        return self.forward_final(x)
