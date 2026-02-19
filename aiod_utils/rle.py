from pathlib import Path
import pickle
from typing import Union, Optional, Any
import warnings

import numpy as np

from aiod_utils.io import reduce_dtype

EXTENSIONS = [".pkl", ".pickle", ".rle"]


def encode(
    mask: np.ndarray,
    mask_type: Optional[str] = None,
    metadata: dict[str, Any] = {},
) -> list[dict]:
    assert isinstance(
        mask, np.ndarray
    ), f"mask must be a numpy array, not {type(mask)}"
    # Convert to lowest bit type
    mask = reduce_dtype(mask)
    # Try to infer the mask type if not provided
    if mask_type is None:
        mask_type = check_mask_type(mask)
        warnings.warn(f"Mask type not provided, inferring as {mask_type}")
    # Give a batch dimension if it's not there
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)
    elif mask.ndim >= 4:
        # First try squeezing in case of pointless dims
        mask = np.squeeze(mask)
        # Raise an error if more than 3D, as we will struggle to guess H & W (& depth)
        if mask.ndim >= 4:
            raise ValueError(
                f"Mask has {mask.ndim} dimensions, must be 2D or 3D (got {mask.shape})"
            )
    if mask_type == "binary":
        # Ensure mask is boolean (rather than 2 unique values, it's faster)
        mask = mask.astype(bool)
        res = _encode_binary(mask)
    elif mask_type == "instance":
        mask = mask.astype(np.int64)
        res = _encode_instance(mask)
    # Insert metadata
    res.append({"metadata": metadata})
    return res


def check_mask_type(mask: np.ndarray) -> str:
    # Boolean masks are binary
    if mask.dtype == bool:
        mask_type = "binary"
    # Masks with only 2 unique values are binary
    elif np.unique(mask).shape[0] <= 2:
        mask_type = "binary"
    # Otherwise, it's an instance mask
    else:
        mask_type = "instance"
    return mask_type


def _encode_binary(mask, **kwargs) -> list[dict]:
    # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/utils/amg.py#L109
    # B here is some kind of batch
    # For binarised instance segmentation, each batch element is a single instance
    # Otherwise each batch element is a slice
    b, h, w = mask.shape
    mask = mask.transpose(0, 2, 1).reshape(b, -1)

    # Compute change indices
    # Essentially, XOR the mask with itself shifted by 1, identifying contiguous regions
    diff = mask[:, 1:] ^ mask[:, :-1]
    # Then find all the indices where we have a change
    change_indices = np.argwhere(diff)

    # Additional metadata
    metadata = {}

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = np.concatenate(
            [
                np.array([0], dtype=cur_idxs.dtype),
                cur_idxs + 1,
                np.array([h * w], dtype=cur_idxs.dtype),
            ]
        )
        # Calculate the run length
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        # Start empty if first pixel is background, otherwise start with 0
        counts = [] if mask[i, 0] == 0 else [0]
        # Convert to list for easier saving
        counts.extend(btw_idxs.tolist())
        # Store the size and RLE/counts and any additional metadata
        if "idx" in kwargs:
            metadata = {"idx": int(kwargs["idx"][i])}
        out.append({"size": [h, w], "counts": counts, **metadata})
    return out


# @line_profiler.profile
def _encode_instance(mask: np.ndarray, **kwargs) -> list[dict]:
    """
    Note that this is much slower than the binary encoding.
    This needs to be used for models like SAM, where instance masks overlap.
    For other instance masks, binary encoding then connected components is likely faster.
    """
    out = []
    #
    # We need to loop over each slice
    for idx in range(mask.shape[0]):
        # Get the mask for this slice
        mask_slice = mask[idx]
        # Get the unique instances
        instances = np.unique(mask_slice)
        # Remove the background
        instances = instances[instances != 0]
        # Handle the case where there are no instances
        if len(instances) == 0:
            mask_batch = np.zeros_like(mask_slice, dtype=bool)[np.newaxis, ...]
            instances = np.array([0], dtype=np.uint8)
        else:
            # Convert into a batch of binarised masks for each instance
            mask_batch = mask_slice[np.newaxis, ...] == np.unique(instances)[:, np.newaxis, np.newaxis]
        # Encode the binary masks
        # Add the instance index to the metadata for later decoding
        encoded_masks = _encode_binary(mask_batch, idx=instances)
        # Store the encoded masks
        out.append(encoded_masks)
    return out


def decode(rle: list[dict], mask_type: Optional[str] = None) -> tuple[np.ndarray, dict]:
    # Try to infer the mask type if not provided
    if mask_type is None:
        mask_type = check_rle_type(rle)
        warnings.warn(f"Mask type not provided, inferring as {mask_type}")
    # TODO: Some basic checks for rle key validity
    # Pop out the metadata
    metadata = rle.pop()
    if mask_type == "binary":
        res = _decode_binary(rle)
    elif mask_type == "instance":
        # TODO: Some additional checks for keys for instance masks?
        res = _decode_instance(rle)
    # NOTE: We squeeze here as any 2D inputs are unsqueezed to 3D for simplicity when encoding
    return res.squeeze(), metadata


def check_rle_type(rle: list[dict]) -> str:
    if isinstance(rle[0], list):
        mask_type = "instance"
    else:
        mask_type = "binary"
    return mask_type


def _decode_binary(rle: list[dict]) -> np.ndarray:
    # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/utils/amg.py#L140
    res = []
    for encoded_mask in rle:
        h, w = encoded_mask["size"]
        mask = np.empty(h * w, dtype=bool)
        idx = 0
        parity = False
        for count in encoded_mask["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            # This acts as a toggle
            parity ^= True
        # Reshape and put in C order (encoded in Fortran order)
        mask = mask.reshape(w, h).transpose()
        res.append(mask)
    return np.stack(res, axis=0, dtype=bool)


def _decode_instance(rle) -> np.ndarray:
    # Container for unknown number of decoded mask slices
    out = []
    # rle_slice is a list of dictionaries for each instance
    for rle_slice in rle:
        # Each dictionary contains the size and counts for the RLE, as well as any metadata
        # NOTE: Could insert the idx at each but the binary version is quicker
        decoded_slice = _decode_binary(rle_slice)
        # Convert stack of binary masks for each instance into a single instance mask
        # Get the instance indices to multiply by the binary masks
        instance_indices = np.array([r["idx"] for r in rle_slice], dtype=np.uint16)
        # Multiply the binary masks by the instance indices and sum to flatten
        decoded_slice = np.einsum("ijk,i->jk", decoded_slice, instance_indices)
        # Append the decoded slice
        out.append(decoded_slice)
    # Reconstruct the full mask array
    return np.stack(out)


def save_encoding(rle: list[dict], fpath: Union[str, Path]):
    # Ensure filename matches
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    # Ensure it's a .pkl file, or other valid extension
    if fpath.suffix not in EXTENSIONS:
        raise ValueError(
            f"Filename cannot have extension {fpath.suffix}, must be one of: {EXTENSIONS}"
        )
    # Save the RLE
    with open(fpath, "wb") as f:
        pickle.dump(rle, f)


def load_encoding(fpath: Union[str, Path]) -> list[dict]:
    # Ensure filename matches
    if not isinstance(fpath, Path):
        fpath = Path(fpath)
    # Cannot load if it doesn't exist
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} does not exist!")
    # Cannot load if not a .pkl file
    if fpath.suffix not in EXTENSIONS:
        raise ValueError(
            f"{fpath} must have an extension in {EXTENSIONS}, not {fpath.suffix}!"
        )
    with open(fpath, "rb") as f:
        return pickle.load(f)


def binary_to_instance(rle):
    # TODO: We could possibly shortcut the conversion and avoid decoding
    mask, metadata = decode(rle, mask_type="binary")
    rle_instance = encode(mask, mask_type="instance", metadata=metadata["metadata"])
    return rle_instance


def instance_to_binary(rle):
    # TODO: We could possibly shortcut the conversion and avoid decoding
    mask, metadata = decode(rle, mask_type="instance")
    rle_binary = encode(mask, mask_type="binary", metadata=metadata["metadata"])
    return rle_binary
