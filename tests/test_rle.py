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
    from aiod_utils.rle import encode, decode

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
    from aiod_utils.rle import encode, decode

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
    from aiod_utils.rle import encode, decode, instance_to_binary

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
    from aiod_utils.rle import encode, decode

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
    """Encoding with no explicit metadata should return an empty metadata dict."""
    mask = request.getfixturevalue(mask)
    from aiod_utils.rle import encode, decode

    rle = encode(mask, mask_type=mask_type)
    _, returned_meta = decode(rle, mask_type=mask_type)

    assert "metadata" in returned_meta
    assert returned_meta["metadata"] == {}


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
    from aiod_utils.rle import encode, decode

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
    from aiod_utils.rle import encode, decode, instance_to_binary

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
    from aiod_utils.rle import encode, decode

    instance_mask = np.array([[0, 1, 1], [3, 0, 0], [0, 2, 0]], dtype=np.uint16)
    # 'idx' is also used internally; passing it here should not silently break things.
    user_meta = {"idx": "custom_value"}

    rle = encode(instance_mask, mask_type="instance", metadata=user_meta)
    decoded_mask, returned_meta = decode(rle, mask_type="instance")

    assert np.array_equal(instance_mask, decoded_mask.astype(instance_mask.dtype))
    assert returned_meta["metadata"] == user_meta


# What other tests should we add?
def test_rle_4d_mask():
    from aiod_utils.rle import encode, decode

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
    from aiod_utils.rle import encode, decode

    mask = request.getfixturevalue(mask)
    # Encode the mask
    rle = encode(mask, mask_type="binary")
    # Decode the mask
    decoded_mask, _ = decode(rle, mask_type="binary")
    # Check that the decoded mask has the same shape as the original mask
    assert decoded_mask.shape == mask.shape
