import numpy as np
import pytest


@pytest.fixture
def binary_2d_mask():
    return np.array([[0, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=np.uint8)


@pytest.fixture
def instance_2d_mask():
    return np.array([[0, 1, 1], [3, 0, 0], [0, 2, 0]], dtype=np.uint16)


@pytest.fixture
def empty_mask():
    return np.zeros((3, 3), dtype=np.uint8)


@pytest.fixture
def single_pixel_mask():
    return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)


@pytest.fixture
def binary_3d_mask():
    return np.array(
        [
            [[0, 1, 1], [1, 0, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[1, 1, 0], [1, 0, 1], [1, 0, 0]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def instance_3d_mask():
    return np.array(
        [
            [[0, 1, 1], [2, 0, 0], [0, 3, 0]],
            [[0, 0, 0], [2, 2, 2], [0, 0, 0]],
            [[4, 4, 0], [4, 0, 2], [5, 0, 0]],
        ],
        dtype=np.uint8,
    )


# Check that encoding and deocding works for 2D & 3D binary masks
@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("binary_2d_mask", "binary"),
        ("instance_2d_mask", "instance"),
        ("binary_3d_mask", "binary"),
        ("instance_3d_mask", "instance"),
    ],
)
def test_rle_encoding_decoding(mask, mask_type, request):
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import decode, encode

    # Encode the mask
    rle = encode(mask, mask_type=mask_type)
    # Decode the mask
    decoded_mask, _ = decode(rle, mask_type=mask_type)
    # Check that the decoded mask matches the original mask
    if mask_type == "binary":
        assert np.array_equal(mask.astype(bool), decoded_mask)
    else:
        assert np.array_equal(mask, decoded_mask.astype(mask.dtype))


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("empty_mask", "binary"),
        ("empty_mask", "instance"),
        ("single_pixel_mask", "binary"),
        ("single_pixel_mask", "instance"),
    ],
)
def test_rle_empty_and_single_pixel_masks(mask, mask_type, request):
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import decode, encode

    # Encode the mask
    rle = encode(mask, mask_type=mask_type)
    # Decode the mask
    decoded_mask, _ = decode(rle, mask_type=mask_type)
    # Check that the decoded mask matches the original mask
    if mask_type == "binary":
        assert np.array_equal(mask.astype(bool), decoded_mask)
    else:
        assert np.array_equal(mask, decoded_mask.astype(mask.dtype))


# NOTE: Conversion to instance currently doesn't work as we don't
# actually do any labelling, and unsure if we want to.
# @pytest.mark.parametrize(
#     "mask, mask_type",
#     [
#         ("binary_2d_mask", "instance_2d_mask"),
#         ("binary_3d_mask", "instance_3d_mask"),
#     ],
# )
# def test_binary_to_instance_conversion(mask, mask_type, request):
#     from aiod_utils.rle import encode, decode, binary_to_instance

#     binary_mask = request.getfixturevalue(mask)
#     instance_mask = request.getfixturevalue(mask_type)

#     # Convert binary mask to instance mask
#     rle_instance = binary_to_instance(encode(binary_mask, mask_type="binary"))
#     # Decode the instance mask
#     decoded_instance_mask, _ = decode(rle_instance, mask_type="instance")
#     breakpoint()
#     # Check that the decoded instance mask matches the original instance mask
#     assert np.array_equal(
#         instance_mask, decoded_instance_mask.astype(instance_mask.dtype)
#     )


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("instance_2d_mask", "binary_2d_mask"),
        ("instance_3d_mask", "binary_3d_mask"),
    ],
)
def test_instance_to_binary_conversion(mask, mask_type, request):
    from aiod_utils.rle import decode, encode, instance_to_binary

    instance_mask = request.getfixturevalue(mask)
    binary_mask = request.getfixturevalue(mask_type)

    # Convert instance mask to binary mask
    rle_binary = instance_to_binary(encode(instance_mask, mask_type="instance"))
    # Decode the binary mask
    decoded_binary_mask, _ = decode(rle_binary, mask_type="binary")

    # Check that the decoded binary mask matches the original binary mask
    assert np.array_equal(binary_mask.astype(bool), decoded_binary_mask)


# ---- Metadata handling tests ----


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("binary_2d_mask", "binary"),
        ("binary_3d_mask", "binary"),
        ("instance_2d_mask", "instance"),
        ("instance_3d_mask", "instance"),
    ],
)
def test_metadata_preserved(mask, mask_type, request):
    """User-supplied metadata must survive an encode → decode round-trip."""
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import decode, encode

    user_meta = {"source": "test_scan", "resolution_mm": 0.5, "labels": [1, 2, 3]}
    rle = encode(mask, mask_type=mask_type, metadata=user_meta)
    _, returned_meta = decode(rle, mask_type=mask_type)

    # decode returns {"metadata": <user_meta>}
    assert "metadata" in returned_meta
    assert returned_meta["metadata"] == user_meta


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("binary_2d_mask", "binary"),
        ("binary_3d_mask", "binary"),
        ("instance_2d_mask", "instance"),
        ("instance_3d_mask", "instance"),
    ],
)
def test_empty_metadata_preserved(mask, mask_type, request):
    """Encoding with no explicit metadata should only contain the auto-inserted mask_type."""
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import decode, encode

    rle = encode(mask, mask_type=mask_type)
    _, returned_meta = decode(rle, mask_type=mask_type)

    assert "metadata" in returned_meta
    assert returned_meta["metadata"] == {"mask_type": mask_type}


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("binary_2d_mask", "binary"),
        ("binary_3d_mask", "binary"),
        ("instance_2d_mask", "instance"),
        ("instance_3d_mask", "instance"),
    ],
)
def test_metadata_does_not_corrupt_mask(mask, mask_type, request):
    """Providing user metadata must not alter the decoded mask data."""
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import decode, encode

    user_meta = {"info": "extra", "value": 42}
    rle_with_meta = encode(mask, mask_type=mask_type, metadata=user_meta)
    rle_without_meta = encode(mask, mask_type=mask_type)

    decoded_with, _ = decode(rle_with_meta, mask_type=mask_type)
    decoded_without, _ = decode(rle_without_meta, mask_type=mask_type)

    assert np.array_equal(decoded_with, decoded_without)


@pytest.mark.parametrize(
    "mask, mask_type",
    [
        ("binary_2d_mask", "binary"),
        ("binary_3d_mask", "binary"),
        ("instance_2d_mask", "instance"),
        ("instance_3d_mask", "instance"),
    ],
)
def test_metadata_not_leaked_into_rle_slices(mask, mask_type, request):
    """User metadata must only appear in the trailing sentinel dict, not in
    any per-slice RLE entry, to avoid polluting the counts/size entries."""
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import encode

    user_meta = {"source": "leak_check"}
    rle = encode(mask, mask_type=mask_type, metadata=user_meta)

    # The last entry is always {"metadata": ...}; everything before it is
    # slice data and must not contain 'metadata' as a top-level key.
    payload = rle[:-1]

    def _has_metadata_key(entry):
        """Recursively check that no dict in entry has a 'metadata' key."""
        if isinstance(entry, dict):
            return "metadata" in entry
        if isinstance(entry, list):
            return any(_has_metadata_key(e) for e in entry)
        return False

    for entry in payload:
        assert not _has_metadata_key(entry), (
            f"Found 'metadata' key in payload slice: {entry}"
        )


def test_metadata_preserved_through_instance_to_binary():
    """instance_to_binary must carry user metadata from the instance RLE
    into the newly-created binary RLE."""
    from aiod_utils.rle import decode, encode, instance_to_binary

    instance_mask = np.array([[0, 1, 1], [3, 0, 0], [0, 2, 0]], dtype=np.uint16)
    user_meta = {"patient_id": "P001", "modality": "CT"}

    rle_instance = encode(instance_mask, mask_type="instance", metadata=user_meta)
    rle_binary = instance_to_binary(rle_instance)
    _, returned_meta = decode(rle_binary, mask_type="binary")

    assert returned_meta["metadata"] == user_meta


def test_metadata_key_collision_with_idx():
    """If a user passes 'idx' inside metadata it must not silently corrupt the
    instance encoding (the internal 'idx' usage in _encode_binary takes a
    numpy array of instance labels, whereas user 'idx' is arbitrary).
    Encoding and decoding must still produce a correct mask."""
    from aiod_utils.rle import decode, encode

    instance_mask = np.array([[0, 1, 1], [3, 0, 0], [0, 2, 0]], dtype=np.uint16)
    # 'idx' is also used internally; passing it here should not silently break things.
    user_meta = {"idx": "custom_value"}

    rle = encode(instance_mask, mask_type="instance", metadata=user_meta)
    decoded_mask, returned_meta = decode(rle, mask_type="instance")

    assert np.array_equal(instance_mask, decoded_mask.astype(instance_mask.dtype))
    assert returned_meta["metadata"] == user_meta


# What other tests should we add?
def test_rle_4d_mask():
    from aiod_utils.rle import encode

    # Create a random 4D binary mask
    mask = np.random.randint(0, 2, (2, 3, 3, 3), dtype=np.uint8)

    # Encode the mask
    with pytest.raises(ValueError):
        # RLE encoding for 4D masks is not implemented
        encode(mask, mask_type="binary")


@pytest.mark.parametrize(
    "mask",
    [
        "binary_2d_mask",
        "instance_2d_mask",
        "binary_3d_mask",
        "instance_3d_mask",
    ],
)
def test_consistent_shape(mask, request):
    from aiod_utils.rle import decode, encode

    mask = request.getfixturevalue(mask)
    # Encode the mask
    rle = encode(mask, mask_type="binary")
    # Decode the mask
    decoded_mask, _ = decode(rle, mask_type="binary")
    # Check that the decoded mask has the same shape as the original mask
    assert decoded_mask.shape == mask.shape


# ---- check_mask_type inference edge cases ----


def test_check_mask_type_binary_zero_one():
    from aiod_utils.rle import check_mask_type

    mask = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    assert check_mask_type(mask) == "binary"


# ---- Scale / sparsity regression coverage ----
# The fixtures above are all tiny (3x3) hand-built arrays. They don't exercise
# instances confined to a subset of Z-slices, fully empty slices, or enough
# volume for run-length math bugs to surface -- add coverage for those here
# as a safety net for the encode/decode performance work.


@pytest.fixture
def sparse_instance_3d_mask():
    """Instances confined to a subset of slices, including one fully empty slice."""
    mask = np.zeros((4, 5, 5), dtype=np.uint8)
    mask[0, 1:3, 1:3] = 1
    # slice 1 intentionally left all-background
    mask[2, 0:2, 0:2] = 2
    mask[2, 3:5, 3:5] = 3
    mask[3, 2:4, 2:4] = 1  # instance 1 reappears in a later, non-contiguous slice
    return mask


def test_sparse_in_z_instance_round_trip(sparse_instance_3d_mask):
    from aiod_utils.rle import decode, encode

    rle = encode(sparse_instance_3d_mask, mask_type="instance")
    decoded_mask, _ = decode(rle, mask_type="instance")
    assert np.array_equal(
        sparse_instance_3d_mask, decoded_mask.astype(sparse_instance_3d_mask.dtype)
    )


def test_large_instance_mask_round_trip():
    """Denser, larger instance mask with uneven per-slice instance membership,
    mimicking real (SAM-like) data -- a regression guard for run-length math
    bugs that only surface at scale, not caught by the small fixtures above."""
    from aiod_utils.rle import decode, encode

    rng = np.random.default_rng(123)
    shape = (24, 64, 64)
    mask = np.zeros(shape, dtype=np.uint16)
    n_instances = 40
    ys, xs = np.ogrid[: shape[1], : shape[2]]
    for instance_id in range(1, n_instances + 1):
        # Each instance only touches a handful of slices, like real data does
        n_slices_for_instance = rng.integers(1, 6)
        slice_idxs = rng.choice(shape[0], size=n_slices_for_instance, replace=False)
        for z in slice_idxs:
            cy, cx = rng.integers(0, shape[1]), rng.integers(0, shape[2])
            radius = rng.integers(2, 6)
            circle = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius**2
            mask[z][circle] = instance_id

    rle = encode(mask, mask_type="instance")
    decoded_mask, _ = decode(rle, mask_type="instance")
    assert np.array_equal(mask, decoded_mask.astype(mask.dtype))


def test_decode_backward_compatible_with_pre_bbox_instance_format():
    """Old-format instance RLE entries (no 'offset'/'full_size', encoded
    against the full frame -- how _encode_instance worked before switching
    to per-instance bounding boxes) must still decode correctly. The
    offset-defaulting/canvas-size logic that makes this work lives in
    _decode_instance (not _decode_binary, which is only used directly for
    the top-level "binary" mask_type)."""
    from aiod_utils.rle import _encode_binary, decode

    mask = np.array([[0, 1, 1], [3, 0, 0], [0, 2, 0]], dtype=np.uint16)
    instances = np.array([1, 2, 3], dtype=np.uint16)
    # Reproduce the pre-bbox _encode_instance behaviour directly: each
    # instance encoded against the *full* frame, no offset/full_size keys.
    mask_batch = mask[np.newaxis, ...] == instances[:, np.newaxis, np.newaxis]
    old_format_slice = _encode_binary(mask_batch, idx=instances)
    rle = [old_format_slice, {"metadata": {"mask_type": "instance"}}]

    decoded_mask, _ = decode(rle, mask_type="instance")
    assert np.array_equal(mask, decoded_mask.astype(mask.dtype))


# ---- Overlapping-instance decode semantics ----
# _encode_instance can't itself produce overlap (one dense label array in,
# one instance per pixel), so these build the RLE by hand -- pinning down
# _decode_instance's "last entry in the list wins" behaviour at genuinely
# overlapping pixels, which replaced the old (never-correct) sum-based
# reconstruction. Real use case per _encode_instance's own docstring: SAM
# masks are allowed to overlap.


def test_decode_overlapping_instances_last_write_wins():
    """Two full-frame (old-format) instances sharing a pixel: the later
    entry in the list wins at the shared pixel; non-overlap pixels are
    unaffected."""
    from aiod_utils.rle import _encode_binary, decode

    mask_a = np.array(
        [[True, False, False], [False, True, False], [False, False, False]]
    )
    mask_b = np.array(
        [[False, False, False], [False, True, False], [False, False, True]]
    )
    entry_a = _encode_binary(mask_a[np.newaxis, ...], idx=[1])[0]
    entry_b = _encode_binary(mask_b[np.newaxis, ...], idx=[2])[0]

    rle = [[entry_a, entry_b], {"metadata": {"mask_type": "instance"}}]
    decoded_mask, _ = decode(rle, mask_type="instance")

    assert decoded_mask[0, 0] == 1
    assert decoded_mask[1, 1] == 2  # shared pixel: entry_b is last, so it wins
    assert decoded_mask[2, 2] == 2
    assert decoded_mask[0, 1] == 0


def test_decode_overlapping_bbox_instances_last_write_wins():
    """Same semantics, but for the new bbox-cropped format with real
    (nonzero) offsets, to exercise the canvas-write path directly."""
    from aiod_utils.rle import _encode_binary, decode

    local_block = np.ones((2, 2), dtype=bool)
    entry_a = _encode_binary(local_block[np.newaxis, ...], idx=[1])[0]
    entry_a["offset"] = [0, 0]
    entry_a["full_size"] = [4, 4]
    entry_b = _encode_binary(local_block[np.newaxis, ...], idx=[2])[0]
    entry_b["offset"] = [1, 1]
    entry_b["full_size"] = [4, 4]

    rle = [[entry_a, entry_b], {"metadata": {"mask_type": "instance"}}]
    decoded_mask, _ = decode(rle, mask_type="instance")

    assert decoded_mask[0, 0] == 1
    assert decoded_mask[0, 1] == 1
    assert decoded_mask[1, 0] == 1
    assert (
        decoded_mask[1, 1] == 2
    )  # overlap: entry_b (bbox rows/cols 1-2) is last, wins
    assert decoded_mask[2, 2] == 2
    assert decoded_mask[0, 2] == 0
    assert decoded_mask[3, 3] == 0


def test_decode_three_way_overlap_last_write_wins():
    """3+-way overlap at one pixel still can't round-trip losslessly (only
    one idx slot per pixel) -- confirm only the last-in-list idx survives."""
    from aiod_utils.rle import _encode_binary, decode

    shared_pixel_mask = np.zeros((3, 3), dtype=bool)
    shared_pixel_mask[1, 1] = True
    entries = [
        _encode_binary(shared_pixel_mask[np.newaxis, ...], idx=[i])[0]
        for i in (5, 6, 7)
    ]

    rle = [entries, {"metadata": {"mask_type": "instance"}}]
    decoded_mask, _ = decode(rle, mask_type="instance")

    assert decoded_mask[1, 1] == 7
    assert np.count_nonzero(decoded_mask) == 1
