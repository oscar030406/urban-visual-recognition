from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RGBIRDepthGateStem(nn.Module):
    """RGB-main stem with IR/depth feature gating.

    Input layout is [R, G, B, IR, Depth]. The RGB branch mirrors YOLO's first
    Conv so pretrained RGB filters can be copied directly. The auxiliary branch
    learns a same-resolution feature gate from IR/depth.
    """

    def __init__(
        self,
        c1: int = 5,
        c2: int = 64,
        k: int = 3,
        s: int = 2,
        p: int | None = None,
        g: int = 1,
        act: bool = True,
        aux_channels: int = 2,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if c1 < 3 + aux_channels:
            raise ValueError(f"expected at least {3 + aux_channels} input channels, got {c1}")
        from ultralytics.nn.modules.conv import Conv, autopad

        self.rgb_channels = 3
        self.aux_channels = aux_channels
        self.rgb_conv = Conv(3, c2, k, s, p, g, act=act)
        padding = autopad(k, p)
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, c2, kernel_size=k, stride=s, padding=padding, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.aux_conv[-1].weight)
        nn.init.zeros_(self.aux_conv[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.rgb_channels + self.aux_channels:
            raise ValueError(
                "RGBIRDepthGateStem expected input layout [R,G,B,IR,Depth], "
                f"got shape {tuple(x.shape)}"
            )
        rgb = x[:, : self.rgb_channels]
        aux = x[:, self.rgb_channels : self.rgb_channels + self.aux_channels]
        rgb_features = self.rgb_conv(rgb)
        gate = torch.sigmoid(self.aux_conv(aux))
        return rgb_features * (1.0 + self.alpha * (gate - 0.5))


def replace_first_conv_with_gate_stem(model, alpha: float = 0.0) -> RGBIRDepthGateStem:
    """Replace YOLO layer 0 with a RGBIRDepthGateStem while preserving metadata."""
    net = model.model if hasattr(model, "model") and hasattr(model.model, "model") else model
    layers = net.model if hasattr(net, "model") else net
    current = layers[0]
    if isinstance(current, RGBIRDepthGateStem):
        current.alpha.data.fill_(float(alpha))
        return current
    if not hasattr(current, "conv"):
        raise TypeError(f"first YOLO layer is not Conv-like: {type(current)!r}")
    conv = current.conv
    kernel = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else int(conv.kernel_size)
    stride = conv.stride[0] if isinstance(conv.stride, tuple) else int(conv.stride)
    padding = conv.padding[0] if isinstance(conv.padding, tuple) else int(conv.padding)
    stem = RGBIRDepthGateStem(
        c1=5,
        c2=conv.out_channels,
        k=kernel,
        s=stride,
        p=padding,
        g=conv.groups,
        act=True,
        alpha=alpha,
    )
    for attr in ("i", "f", "type"):
        if hasattr(current, attr):
            setattr(stem, attr, getattr(current, attr))
    stem.type = f"{RGBIRDepthGateStem.__module__}.{RGBIRDepthGateStem.__name__}"
    stem.np = sum(p.numel() for p in stem.parameters())
    layers[0] = stem
    return stem


def copy_gate_stem_rgb_weights_from_model(model, source_model) -> int:
    """Copy source YOLO first-Conv RGB weights into RGBIRDepthGateStem."""
    net = model.model if hasattr(model, "model") and hasattr(model.model, "model") else model
    layers = net.model if hasattr(net, "model") else net
    stem = layers[0]
    if not isinstance(stem, RGBIRDepthGateStem):
        return 0

    if source_model is None or not hasattr(source_model, "state_dict"):
        return 0

    source_state = source_model.float().state_dict()
    target_state = model.state_dict() if hasattr(model, "state_dict") else net.state_dict()
    key_map = {
        "model.0.conv.weight": "model.0.rgb_conv.conv.weight",
        "model.0.bn.weight": "model.0.rgb_conv.bn.weight",
        "model.0.bn.bias": "model.0.rgb_conv.bn.bias",
        "model.0.bn.running_mean": "model.0.rgb_conv.bn.running_mean",
        "model.0.bn.running_var": "model.0.rgb_conv.bn.running_var",
        "model.0.bn.num_batches_tracked": "model.0.rgb_conv.bn.num_batches_tracked",
    }
    copied = {}
    for source_key, target_key in key_map.items():
        source_value = source_state.get(source_key)
        target_value = target_state.get(target_key)
        if source_value is None or target_value is None:
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            continue
        copied[target_key] = source_value.to(dtype=target_value.dtype)
    if not copied:
        return 0
    target_state.update(copied)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(target_state, strict=False)
    else:
        net.load_state_dict(target_state, strict=False)
    return len(copied)


def load_gate_stem_rgb_weights(model, checkpoint: str) -> int:
    """Load a checkpoint and copy its first-Conv RGB weights into RGBIRDepthGateStem."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_model = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else None
    return copy_gate_stem_rgb_weights_from_model(model, source_model)


class ECA(nn.Module):
    """Efficient channel attention with residual gating for YOLO feature maps."""

    def __init__(self, kernel_size: int = 3, alpha: float = 0.0) -> None:
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1).sigmoid()
        return x * (1.0 + self.alpha * weights)


class CSA(nn.Module):
    """Lightweight channel-spatial attention with residual gating."""

    def __init__(
        self,
        channel_kernel_size: int = 3,
        spatial_kernel_size: int = 7,
        alpha: float = 0.0,
    ) -> None:
        super().__init__()
        if channel_kernel_size % 2 == 0 or channel_kernel_size < 1:
            raise ValueError("channel_kernel_size must be a positive odd integer")
        if spatial_kernel_size % 2 == 0 or spatial_kernel_size < 1:
            raise ValueError("spatial_kernel_size must be a positive odd integer")
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=channel_kernel_size,
            padding=(channel_kernel_size - 1) // 2,
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=spatial_kernel_size,
            padding=(spatial_kernel_size - 1) // 2,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = self.channel_pool(x).squeeze(-1).transpose(-1, -2)
        channel = self.channel_conv(channel).transpose(-1, -2).unsqueeze(-1).sigmoid()
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        spatial = self.spatial_conv(torch.cat((avg_map, max_map), dim=1)).sigmoid()
        return x * (1.0 + self.alpha * channel * spatial)


class AuxFeatureGate(nn.Module):
    """Gate one RGB feature map with raw IR/depth auxiliary channels."""

    def __init__(self, out_channels: int, hidden_channels: int = 64, alpha: float = 0.1) -> None:
        super().__init__()
        hidden = max(8, min(int(hidden_channels), int(out_channels) // 4))
        self.proj = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, aux: torch.Tensor, rgb_feature: torch.Tensor) -> torch.Tensor:
        aux = F.interpolate(aux, size=rgb_feature.shape[-2:], mode="bilinear", align_corners=False)
        param = next(self.proj.parameters())
        aux = aux.to(dtype=param.dtype, device=param.device)
        gate = torch.sigmoid(self.proj(aux)).to(dtype=rgb_feature.dtype, device=rgb_feature.device)
        return rgb_feature * (1.0 + self.alpha * (gate - 0.5))


class AuxPyramidGates(nn.Module):
    """P3/P4/P5 IR-depth gates attached to an RGB-main YOLO model via hooks."""

    def __init__(
        self,
        layer_channels: dict[int, int] | None = None,
        hidden_channels: int = 64,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.layer_channels = layer_channels or {16: 256, 19: 512, 22: 512}
        self.gates = nn.ModuleDict(
            {
                str(layer): AuxFeatureGate(channels, hidden_channels=hidden_channels, alpha=alpha)
                for layer, channels in self.layer_channels.items()
            }
        )
        self._aux: torch.Tensor | None = None

    def capture_rgb_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 5:
            raise ValueError(f"AuxPyramidGates expected 5-channel input [RGB,IR,Depth], got {tuple(x.shape)}")
        self._aux = x[:, 3:5]
        return x[:, :3]

    def gate_feature(self, layer: int, feature: torch.Tensor) -> torch.Tensor:
        if self._aux is None:
            raise RuntimeError("AuxPyramidGates did not capture an auxiliary input before feature gating")
        return self.gates[str(layer)](self._aux, feature)


class AuxInputPreHook:
    """Pickle-safe pre-hook that slices 5-channel input to RGB."""

    def __init__(self, gates: AuxPyramidGates) -> None:
        self.gates = gates

    def __call__(self, _module, inputs):
        x = inputs[0]
        return (self.gates.capture_rgb_input(x), *inputs[1:])


class AuxGateForwardHook:
    """Pickle-safe forward hook that gates one pyramid feature."""

    def __init__(self, gates: AuxPyramidGates, layer: int) -> None:
        self.gates = gates
        self.layer = int(layer)

    def __call__(self, _module, _inputs, output):
        return self.gates.gate_feature(self.layer, output)


def _resolve_yolo_layers(model):
    net = model.model if hasattr(model, "model") and hasattr(model.model, "model") else model
    layers = net.model if hasattr(net, "model") else net
    return net, layers


def attach_aux_pyramid_gates(
    model,
    layer_channels: dict[int, int] | None = None,
    hidden_channels: int = 64,
    alpha: float = 0.1,
) -> AuxPyramidGates:
    """Attach P3/P4/P5 IR-depth feature gates to a YOLO DetectionModel.

    The original YOLO model remains RGB-main: a pre-hook slices the incoming
    5-channel tensor to RGB before layer 0, while forward hooks inject IR/depth
    gates at selected pyramid layers.
    """
    net, layers = _resolve_yolo_layers(model)
    layer_channels = layer_channels or {16: 256, 19: 512, 22: 512}
    if len(layers) <= max(layer_channels):
        raise ValueError(f"model has {len(layers)} layers, cannot gate layers {sorted(layer_channels)}")

    for module in [layers[0], *(layers[layer] for layer in sorted(layer_channels))]:
        for hook_id, hook in list(module._forward_pre_hooks.items()):
            if isinstance(hook, AuxInputPreHook):
                del module._forward_pre_hooks[hook_id]
        for hook_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, AuxGateForwardHook):
                del module._forward_hooks[hook_id]

    existing = getattr(net, "aux_pyramid_gates", None)
    if isinstance(existing, AuxPyramidGates) and set(existing.layer_channels) == set(layer_channels):
        gates = existing
    else:
        gates = AuxPyramidGates(layer_channels=layer_channels, hidden_channels=hidden_channels, alpha=alpha)
        net.add_module("aux_pyramid_gates", gates)

    layers[0].register_forward_pre_hook(AuxInputPreHook(gates))
    for layer in sorted(layer_channels):
        layers[layer].register_forward_hook(AuxGateForwardHook(gates, layer))
    return gates


def register_ultralytics_attention() -> None:
    """Expose local attention modules to Ultralytics YAML parsing."""
    from ultralytics.nn import tasks

    tasks.ECA = ECA
    tasks.CSA = CSA
    tasks.RGBIRDepthGateStem = RGBIRDepthGateStem
    tasks.AuxFeatureGate = AuxFeatureGate
    tasks.AuxPyramidGates = AuxPyramidGates
    tasks.AuxInputPreHook = AuxInputPreHook
    tasks.AuxGateForwardHook = AuxGateForwardHook
