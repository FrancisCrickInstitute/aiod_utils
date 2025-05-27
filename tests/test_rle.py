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
            [[1, 0, 0], [0, 1, 1], [1, 0, 0]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def instance_3d_mask():
    return np.array(
        [
            [[0, 1, 1], [3, 0, 0], [0, 2, 0]],
            [[0, 0, 0], [4, 4, 4], [0, 0, 0]],
            [[5, 0, 0], [0, 6, 6], [5, 0, 0]],
        ],
        dtype=np.uint8,
    )


# @pytest.fixture
# def binary_4d_mask():
#     return np.random.randint(0, 2, (2, 3, 3, 3), dtype=np.uint8)


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


# What other tests should we add?
def test_rle_4d_mask():
    from aiod_utils.rle import encode, decode

    # Create a random 4D binary mask
    mask = np.random.randint(0, 2, (2, 3, 3, 3), dtype=np.uint8)

    # Encode the mask
    with pytest.raises(ValueError):
        # RLE encoding for 4D masks is not implemented
        encode(mask, mask_type="binary")
