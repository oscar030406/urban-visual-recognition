from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only on minimal environments.
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for file image IO. Install opencv-python.")


def read_image(path: Path, unchanged: bool = False) -> np.ndarray:
    """Read an image with OpenCV while preserving 16-bit depth when requested."""
    _require_cv2()
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    buffer = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(buffer, flag)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"failed to write image: {path}")
    encoded.tofile(path)


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert a 2D or 3D image array to one uint8 channel."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.mean(image[..., :3], axis=2)
    else:
        raise ValueError(f"expected 2D or 3D image, got shape {image.shape}")

    if gray.dtype == np.uint8:
        return gray
    gray = gray.astype(np.float32)
    max_value = float(np.nanmax(gray)) if gray.size else 0.0
    if max_value <= 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    return np.clip(gray / max_value * 255.0, 0, 255).astype(np.uint8)


def enhance_infrared(image: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """Return a single-channel infrared image, optionally with CLAHE contrast enhancement."""
    gray = to_gray_uint8(image)
    if not use_clahe:
        return gray
    _require_cv2()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize_depth_to_uint8(
    depth: np.ndarray,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize depth to uint8 and return a valid-depth mask.

    Official depth PNG files are 16-bit millimeter maps, but the released data also
    contains 8-bit JPG depth maps. Treating those JPG values as millimeters collapses
    most of the depth channel to zero, so uint8 depth is preserved as an image-space
    depth prior while uint16 depth keeps the millimeter normalization.
    """
    if depth.ndim == 3:
        depth = depth[..., 0]
    if min_depth >= max_depth:
        raise ValueError("min_depth must be smaller than max_depth")

    finite_depth = depth[np.isfinite(depth)] if np.issubdtype(depth.dtype, np.floating) else depth
    observed_max = float(np.nanmax(finite_depth)) if finite_depth.size else 0.0
    if depth.dtype == np.uint8 or observed_max <= 255.0:
        normalized = to_gray_uint8(depth)
        valid_mask = normalized > 0
        return normalized, valid_mask

    depth_float = depth.astype(np.float32)
    valid_mask = depth_float >= float(min_depth)
    clipped = np.clip(depth_float, float(min_depth), float(max_depth))
    normalized = (clipped - float(min_depth)) / float(max_depth - min_depth)
    normalized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    normalized[~valid_mask] = 0
    return normalized, valid_mask


def rgb_luma(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB-like input to a luminance channel without assuming color semantics."""
    if image.ndim == 2:
        return to_gray_uint8(image)
    if image.shape[2] < 3:
        raise ValueError(f"expected at least 3 channels, got shape {image.shape}")
    channels = image[..., :3].astype(np.float32)
    luma = 0.114 * channels[..., 0] + 0.587 * channels[..., 1] + 0.299 * channels[..., 2]
    return np.clip(luma, 0, 255).astype(np.uint8)


def make_triad_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """Build a YOLO-compatible 3-channel fusion image: RGB luma, enhanced IR, normalized depth."""
    rgb_channel = rgb_luma(rgb)
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    depth_channel, _ = normalize_depth_to_uint8(depth, min_depth=min_depth, max_depth=max_depth)

    if rgb_channel.shape != ir_channel.shape or rgb_channel.shape != depth_channel.shape:
        raise ValueError(
            "modality shapes must match: "
            f"rgb={rgb_channel.shape}, infrared={ir_channel.shape}, depth={depth_channel.shape}"
        )
    return np.dstack([rgb_channel, ir_channel, depth_channel]).astype(np.uint8)


def _normalize_float01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    min_value = float(np.nanmin(values)) if values.size else 0.0
    max_value = float(np.nanmax(values)) if values.size else 0.0
    if max_value <= min_value:
        return np.zeros(values.shape, dtype=np.float32)
    return (values - min_value) / (max_value - min_value)


def _require_color_image(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected color image with at least 3 channels, got shape {rgb.shape}")
    return rgb[..., :3]


def _check_spatial_match(rgb: np.ndarray, *channels: np.ndarray) -> None:
    for channel in channels:
        if rgb.shape[:2] != channel.shape[:2]:
            raise ValueError(
                "modality shapes must match: "
                f"rgb={rgb.shape[:2]}, other={channel.shape[:2]}"
            )


def stack_single_channel(channel: np.ndarray) -> np.ndarray:
    """Build a 3-channel uint8 image from one channel."""
    channel = to_gray_uint8(channel)
    return np.dstack([channel, channel, channel]).astype(np.uint8)


def make_infrared_image(infrared: np.ndarray, use_clahe: bool = True) -> np.ndarray:
    """Build a YOLO-compatible infrared-only diagnostic image."""
    return stack_single_channel(enhance_infrared(infrared, use_clahe=use_clahe))


def make_depth_image(
    depth: np.ndarray,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """Build a YOLO-compatible depth-only diagnostic image."""
    depth_channel, _ = normalize_depth_to_uint8(depth, min_depth=min_depth, max_depth=max_depth)
    return stack_single_channel(depth_channel)


def _smooth_attention(attention: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return attention
    min_side = min(attention.shape[:2])
    if min_side >= 5:
        return cv2.GaussianBlur(attention, (5, 5), 0)
    if min_side >= 3:
        return cv2.GaussianBlur(attention, (3, 3), 0)
    return attention


def _apply_rgb_attention(
    rgb: np.ndarray,
    attention: np.ndarray,
    base_gate: float,
    gain: float,
) -> np.ndarray:
    rgb = _require_color_image(rgb)
    _check_spatial_match(rgb, attention)
    attention = _smooth_attention(np.clip(attention.astype(np.float32), 0.0, 1.0))
    gate = np.clip(base_gate + gain * attention, 0.0, 2.0)
    guided = rgb.astype(np.float32) * gate[..., None]
    return np.clip(guided, 0, 255).astype(np.uint8)


def make_rgb_guided_ir_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    use_clahe: bool = True,
    base_gate: float = 0.90,
    gain: float = 0.25,
) -> np.ndarray:
    """Preserve RGB while using infrared saliency as a spatial gain map."""
    rgb = _require_color_image(rgb)
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    _check_spatial_match(rgb, ir_channel)
    ir_attention = _normalize_float01(ir_channel)
    return _apply_rgb_attention(rgb, ir_attention, base_gate=base_gate, gain=gain)


def make_rgb_guided_depth_image(
    rgb: np.ndarray,
    depth: np.ndarray,
    min_depth: int = 100,
    max_depth: int = 20_000,
    base_gate: float = 0.90,
    gain: float = 0.25,
) -> np.ndarray:
    """Preserve RGB while using valid near-depth saliency as a spatial gain map."""
    rgb = _require_color_image(rgb)
    depth_channel, valid_mask = normalize_depth_to_uint8(
        depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    _check_spatial_match(rgb, depth_channel)
    near_depth = (1.0 - depth_channel.astype(np.float32) / 255.0) * valid_mask.astype(np.float32)
    depth_attention = _normalize_float01(near_depth)
    return _apply_rgb_attention(rgb, depth_attention, base_gate=base_gate, gain=gain)


def make_rgb_guided_rdt_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
    ir_weight: float = 0.55,
    depth_weight: float = 0.45,
    base_gate: float = 0.85,
    gain: float = 0.30,
) -> np.ndarray:
    """Preserve RGB distribution while IR and depth provide spatial guidance.

    The output remains a normal 3-channel color image, so COCO-pretrained YOLO
    filters keep seeing RGB-like inputs. Infrared and near-depth signals only
    modulate local brightness instead of replacing color channels.
    """
    rgb = _require_color_image(rgb)
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    depth_channel, valid_mask = normalize_depth_to_uint8(
        depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    _check_spatial_match(rgb, ir_channel, depth_channel)

    weight_sum = ir_weight + depth_weight
    if weight_sum <= 0:
        raise ValueError("ir_weight + depth_weight must be positive")

    ir_attention = _normalize_float01(ir_channel)
    near_depth = (1.0 - depth_channel.astype(np.float32) / 255.0) * valid_mask.astype(np.float32)
    depth_attention = _normalize_float01(near_depth)
    attention = (ir_weight * ir_attention + depth_weight * depth_attention) / weight_sum
    return _apply_rgb_attention(rgb, attention, base_gate=base_gate, gain=gain)


def make_rgb_guided_rdt_v2_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """A conservative RGB-guided RDT variant with weaker IR/depth gain."""
    return make_rgb_guided_rdt_image(
        rgb=rgb,
        infrared=infrared,
        depth=depth,
        use_clahe=use_clahe,
        min_depth=min_depth,
        max_depth=max_depth,
        ir_weight=0.55,
        depth_weight=0.45,
        base_gate=0.90,
        gain=0.20,
    )


def make_rgb_guided_rdt_v3_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """A depth-heavier RGB-guided RDT variant for geometry-sensitive boxes."""
    return make_rgb_guided_rdt_image(
        rgb=rgb,
        infrared=infrared,
        depth=depth,
        use_clahe=use_clahe,
        min_depth=min_depth,
        max_depth=max_depth,
        ir_weight=0.45,
        depth_weight=0.55,
        base_gate=0.85,
        gain=0.30,
    )


def make_cssa_lite_image(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """Build a CSSA-inspired 3-channel image with IR/depth spatial attention.

    This is not a reimplementation of the paper's network module. It is a low-risk
    preprocessing variant that preserves YOLO compatibility while injecting a spatial
    saliency prior from thermal and valid near-depth regions.
    """
    rgb_channel = rgb_luma(rgb)
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    depth_channel, valid_mask = normalize_depth_to_uint8(
        depth,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    if rgb_channel.shape != ir_channel.shape or rgb_channel.shape != depth_channel.shape:
        raise ValueError(
            "modality shapes must match: "
            f"rgb={rgb_channel.shape}, infrared={ir_channel.shape}, depth={depth_channel.shape}"
        )

    ir_attention = _normalize_float01(ir_channel)
    near_depth = (1.0 - depth_channel.astype(np.float32) / 255.0) * valid_mask.astype(np.float32)
    depth_attention = _normalize_float01(near_depth)
    attention = np.clip(0.6 * ir_attention + 0.4 * depth_attention, 0.0, 1.0)
    if cv2 is not None and min(attention.shape[:2]) >= 3:
        attention = cv2.GaussianBlur(attention, (3, 3), 0)

    gate = 0.75 + 0.50 * attention
    attended_rgb = np.clip(rgb_channel.astype(np.float32) * gate, 0, 255).astype(np.uint8)
    return np.dstack([attended_rgb, ir_channel, depth_channel]).astype(np.uint8)


def apply_modality_dropout(image: np.ndarray, channels: list[int] | tuple[int, ...]) -> np.ndarray:
    """Zero selected channels in a fused image for modality-dropout augmentation."""
    if image.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {image.shape}")
    output = image.copy()
    for channel in channels:
        if channel < 0 or channel >= output.shape[2]:
            raise ValueError(f"channel index out of range: {channel}")
        output[..., channel] = 0
    return output


def make_rgb_ir_depth_gate_array(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """Build a 5-channel uint8 image for RGB-main IR/depth-gated models.

    The first three channels are RGB-ordered color channels because Ultralytics
    only applies BGR-to-RGB reordering to normal 3-channel images. The final two
    channels are enhanced infrared and normalized depth.
    """
    rgb = _require_color_image(rgb)
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    depth_channel, _ = normalize_depth_to_uint8(depth, min_depth=min_depth, max_depth=max_depth)
    _check_spatial_match(rgb, ir_channel, depth_channel)
    rgb_order = rgb[..., :3][..., ::-1]
    return np.dstack([rgb_order, ir_channel, depth_channel]).astype(np.uint8)


def make_rgb_guided_rdt_gate_array(
    rgb: np.ndarray,
    infrared: np.ndarray,
    depth: np.ndarray,
    use_clahe: bool = True,
    min_depth: int = 100,
    max_depth: int = 20_000,
) -> np.ndarray:
    """Build 5-channel input with rgb_guided_rdt as RGB-main channels.

    The first three channels match the RGB-like distribution used by the locked
    best rgb_guided_rdt detector, while IR and depth remain available as raw
    auxiliary gate channels.
    """
    guided = make_rgb_guided_rdt_image(
        rgb=rgb,
        infrared=infrared,
        depth=depth,
        use_clahe=use_clahe,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    ir_channel = enhance_infrared(infrared, use_clahe=use_clahe)
    depth_channel, _ = normalize_depth_to_uint8(depth, min_depth=min_depth, max_depth=max_depth)
    _check_spatial_match(guided, ir_channel, depth_channel)
    guided_rgb_order = guided[..., :3][..., ::-1]
    return np.dstack([guided_rgb_order, ir_channel, depth_channel]).astype(np.uint8)
