import numpy as np

from city_multimodal_detection.image_ops import (
    apply_modality_dropout,
    enhance_infrared,
    make_cssa_lite_image,
    make_rgb_guided_rdt_image,
    make_triad_image,
    normalize_depth_to_uint8,
)


def test_normalize_depth_clamps_invalid_and_far_values():
    depth = np.array(
        [
            [0, 100, 1_000],
            [20_000, 25_000, 5_000],
        ],
        dtype=np.uint16,
    )

    normalized, valid_mask = normalize_depth_to_uint8(depth, min_depth=100, max_depth=20_000)

    assert normalized.dtype == np.uint8
    assert valid_mask.dtype == bool
    assert normalized[0, 0] == 0
    assert normalized[0, 1] == 0
    assert normalized[1, 1] == 255
    assert valid_mask.tolist() == [[False, True, True], [True, True, True]]


def test_normalize_depth_preserves_uint8_depth_maps():
    depth = np.array(
        [
            [0, 20, 180],
            [255, 100, 5],
        ],
        dtype=np.uint8,
    )

    normalized, valid_mask = normalize_depth_to_uint8(depth, min_depth=100, max_depth=20_000)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == depth.tolist()
    assert valid_mask.tolist() == [[False, True, True], [True, True, True]]


def test_enhance_infrared_returns_single_channel_uint8():
    infrared = np.dstack(
        [
            np.arange(9, dtype=np.uint8).reshape(3, 3),
            np.arange(9, dtype=np.uint8).reshape(3, 3),
            np.arange(9, dtype=np.uint8).reshape(3, 3),
        ]
    )

    enhanced = enhance_infrared(infrared, use_clahe=False)

    assert enhanced.shape == (3, 3)
    assert enhanced.dtype == np.uint8


def test_make_triad_image_uses_rgb_luma_infrared_and_depth_channels():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    infrared = np.full((2, 2, 3), 80, dtype=np.uint8)
    depth = np.full((2, 2), 2_000, dtype=np.uint16)

    triad = make_triad_image(rgb, infrared, depth, use_clahe=False)

    assert triad.shape == (2, 2, 3)
    assert triad.dtype == np.uint8
    assert triad[..., 0].mean() > 0
    assert np.all(triad[..., 1] == 80)
    assert triad[..., 2].mean() > 0


def test_make_cssa_lite_image_enhances_salient_ir_depth_regions():
    rgb = np.full((4, 4, 3), 30, dtype=np.uint8)
    infrared = np.zeros((4, 4, 3), dtype=np.uint8)
    infrared[1:3, 1:3] = 220
    depth = np.full((4, 4), 10_000, dtype=np.uint16)
    depth[1:3, 1:3] = 1_000

    fused = make_cssa_lite_image(rgb, infrared, depth, use_clahe=False)

    assert fused.shape == (4, 4, 3)
    assert fused.dtype == np.uint8
    assert fused[1:3, 1:3, 0].mean() > fused[[0, 0, 3, 3], [0, 3, 0, 3], 0].mean()


def test_make_rgb_guided_rdt_image_preserves_color_channels():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 90
    rgb[..., 2] = 150
    infrared = np.zeros((4, 4, 3), dtype=np.uint8)
    infrared[1:3, 1:3] = 220
    depth = np.full((4, 4), 10_000, dtype=np.uint16)
    depth[1:3, 1:3] = 1_000

    guided = make_rgb_guided_rdt_image(rgb, infrared, depth, use_clahe=False)

    assert guided.shape == (4, 4, 3)
    assert guided.dtype == np.uint8
    assert not np.array_equal(guided[..., 0], guided[..., 1])
    assert not np.array_equal(guided[..., 1], guided[..., 2])
    assert guided[1:3, 1:3].mean() > guided[[0, 0, 3, 3], [0, 3, 0, 3]].mean()


def test_apply_modality_dropout_zeros_selected_channels():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)

    dropped = apply_modality_dropout(image, channels=[1, 2])

    assert np.all(dropped[..., 0] == 100)
    assert np.all(dropped[..., 1] == 0)
    assert np.all(dropped[..., 2] == 0)
