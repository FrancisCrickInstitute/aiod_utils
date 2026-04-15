import pytest
from numpy import dtype as np_dtype

from aiod_utils.stacks import (
    HEADROOM_FACTOR,
    MAX_SUBSTACK_SIZE,
    Stack,
    auto_size,
    calc_num_stacks,
    calc_num_stacks_dim,
    compute_max_substack_size,
)

# https://docs.seqera.io/nextflow/reference/stdlib-types#memoryunit
MiB = 2**20
GiB = 2**30


# ── helpers ───────────────────────────────────────────────────────────────────

def max_voxels(memory_bytes, dtype, n_channels):
    return memory_bytes * HEADROOM_FACTOR / (np_dtype(dtype).itemsize * n_channels)


def uniform_cap(val):
    """Stack used as a cap with identical value in all spatial dims."""
    return Stack(height=val, width=val, depth=val, channels=1_000_000)


# ── auto_size ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size, max_size, expected", [
    (1000, 200,  5),   # exact fit
    (1001, 200,  6),   # just over → extra split
    (100,  200,  1),   # image smaller than cap → 1 split
    (200,  200,  1),   # exact cap size → 1 split
    (1,    200,  1),   # single pixel
])
def test_auto_size(size, max_size, expected):
    assert auto_size(size, max_size) == expected


# ── compute_max_substack_size ─────────────────────────────────────────────────

@pytest.mark.parametrize("memory_bytes, dtype, n_channels, img_shape", [
    (100 * GiB, "uint8",   1, Stack(500,  500,  50,  1)),  # image fits easily
    (100 * GiB, "float32", 1, Stack(200,  200,  20,  1)),  # image fits easily
    (100 * MiB, "float32", 1, Stack(2000, 400, 100,  1)),  # image must be split
    (100 * MiB, "float32", 3, Stack(2000, 400, 100,  3)),  # multi-channel, split
])
def test_compute_max_substack_size_budget_not_exceeded(memory_bytes, dtype, n_channels, img_shape):
    result = compute_max_substack_size(memory_bytes, dtype, img_shape)
    voxels = result.height * result.width * result.depth
    assert voxels <= max_voxels(memory_bytes, dtype, n_channels)


@pytest.mark.parametrize("memory_bytes, dtype, n_channels, img_shape", [
    (100 * GiB, "uint8",   1, Stack(500, 500, 50, 1)),
    (100 * GiB, "float32", 1, Stack(200, 200, 20, 1)),
])
def test_compute_max_substack_size_whole_image_fits(memory_bytes, dtype, n_channels, img_shape):
    result = compute_max_substack_size(memory_bytes, dtype, img_shape)
    assert result.height == img_shape.height
    assert result.width  == img_shape.width
    assert result.depth  == img_shape.depth


def test_compute_max_substack_size_degenerate_budget():
    """Sub-voxel budget should raise ValueError — this is a misconfiguration."""
    with pytest.raises(ValueError, match="Memory budget too small"):
        compute_max_substack_size(1, "float32", Stack(500, 500, 50, 1))


# Anisotropic image: H:W = 5:1, H:D = 20:1 — meaningful aspect ratios to verify.
# At 100MB float32 this produces h≈1252, w≈250, d≈62 (computed analytically).
ANISO = Stack(2000, 400, 100, 1)

def test_compute_max_substack_size_aspect_ratio_hw():
    result = compute_max_substack_size(100 * MiB, "float32", ANISO)
    assert result.height / result.width == pytest.approx(ANISO.height / ANISO.width, rel=0.05)


def test_compute_max_substack_size_aspect_ratio_hd():
    result = compute_max_substack_size(100 * MiB, "float32", ANISO)
    assert result.height / result.depth == pytest.approx(ANISO.height / ANISO.depth, rel=0.05)


def test_compute_max_substack_size_channels_field():
    """channels in result == n_channels, regardless of image_shape.channels."""
    result = compute_max_substack_size(1 * GiB, "float32", Stack(500, 500, 50, 3))
    assert result.channels == 3


def test_compute_max_substack_size_multichannel_smaller():
    """More channels → larger bytes/voxel → smaller substack for same memory."""
    img = Stack(10000, 10000, 200, 1)
    single = compute_max_substack_size(1 * GiB, "float32", img)
    img = Stack(10000, 10000, 200, 3)
    multi  = compute_max_substack_size(1 * GiB, "float32", img)
    vol_single = single.height * single.width * single.depth
    vol_multi  = multi.height  * multi.width  * multi.depth
    assert vol_multi < vol_single


# ── calc_num_stacks_dim ───────────────────────────────────────────────────────
#
# All cases use overlap=0.0 (eff_size == dim_size) unless noted.
# cap is a uniform Stack so any dim attribute gives the same value.

@pytest.mark.parametrize("dim_size, req, cap_val, expected_n", [
    # auto mode
    (1000, "auto", 200,  5),   # 1000//200=5, 1000/5==200 (exact) → 5
    (1001, "auto", 200,  6),   # 1001//200=5, 1001/5>200 → 6
    (100,  "auto", 200,  1),   # image smaller than cap → 1
    # explicit valid request (no sanity override triggered)
    (1000, "3",    200,  3),
    (1000, "1",    500,  1),   # 1 substack, substack=1000 < 500*3=1500 and > 500/20=25 ✓
    # clamp: req < 1 → max(1, req) = 1
    (100,  "0",    200,  1),
    # override: more splits than pixels
    (50,   "200",  200,  1),   # 200 > 50 → auto_size(50,200)=1
    # override: substack too small (< cap/20)
    # dim=1000, req=30, cap=1000 → substack=33 < 50 → auto_size(1000,1000)=1
    (1000, "30",  1000,  1),
    # override: substack too large (> cap*3)
    # dim=1000, req=2, cap=10 → substack=500 > 30 → auto_size(1000,10)=100
    (1000, "2",    10,  100),
    # explicit req=1 where single substack would exceed cap*3
    # dim=1000, req=1, cap=200 → substack=1000 > 600 → auto_size(1000,200)=5
    (1000, "1",   200,   5),
])
def test_calc_num_stacks_dim(dim_size, req, cap_val, expected_n):
    n, eff = calc_num_stacks_dim(dim_size, req, 0.0, "height", uniform_cap(cap_val))
    assert n == expected_n
    assert eff == dim_size  # overlap=0 so eff_size == dim_size


def test_calc_num_stacks_dim_overlap_inflates_eff_size():
    n, eff = calc_num_stacks_dim(1000, "auto", 0.1, "height", uniform_cap(200))
    assert eff == round(1000 * 1.1)   # eff_size correctly inflated
    assert n >= 1


def test_calc_num_stacks_dim_overlap_warns_on_single_stack(recwarn):
    calc_num_stacks_dim(100, "1", 0.1, "height", uniform_cap(200))
    assert any("overlap" in str(w.message).lower() for w in recwarn.list)


# ── calc_num_stacks: channels never split ─────────────────────────────────────

@pytest.mark.parametrize("channels", [1, 3, 16])
def test_calc_num_stacks_channels_not_split(channels):
    img = Stack(1000, 1000, 50, channels)
    req = Stack("auto", "auto", "auto", "auto")
    overlap = Stack(0.0, 0.0, 0.0, 0.0)
    num, _ = calc_num_stacks(img, req, overlap)
    assert num.channels == channels


# ── end-to-end: more memory → fewer or equal substacks ───────────────────────

def test_more_memory_fewer_substacks():
    img = Stack(5000, 5000, 100, 1)
    req = Stack("auto", "auto", "auto", "auto")
    overlap = Stack(0.0, 0.0, 0.0, 0.0)

    prev_total = None
    for memory_gb in [1, 4, 16, 64]:
        cap = compute_max_substack_size(memory_gb * GiB, "float32", img)
        num, _ = calc_num_stacks(img, req, overlap, cap)
        total = num.height * num.width * num.depth
        if prev_total is not None:
            assert total <= prev_total, (
                f"Doubling memory should not increase substack count "
                f"(was {prev_total} at {memory_gb//2}GB, got {total} at {memory_gb}GB)"
            )
        prev_total = total