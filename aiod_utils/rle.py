import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from aiod_utils.io import reduce_dtype

EXTENSIONS = [".pkl", ".pickle", ".rle"]


def encode(
    mask: np.ndarray,
    mask_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict]:
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy array, not {type(mask)}")
    if metadata is None:
        metadata = {}
    # Convert to lowest bit type
    mask = reduce_dtype(mask)
    # Try to infer the mask type if not provided
    if mask_type is None:
        mask_type = check_mask_type(mask)
        warnings.warn(f"Mask type not provided, inferring as {mask_type}", stacklevel=2)
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
        res = _encode_instance(mask, **metadata)
    # Store mask_type in metadata for self-documentation
    metadata["mask_type"] = mask_type
    # Insert metadata
    res.append({"metadata": metadata})
    return res


def check_mask_type(mask: np.ndarray) -> str:
    # Boolean masks are binary
    if mask.dtype == bool or np.unique(mask).shape[0] <= 2:
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

    # np.argwhere on a 2D array returns rows already sorted by batch index
    # then column index, so split once into per-batch-element groups instead
    # of re-scanning the whole array with a fresh boolean mask for every `i`
    # (that was O(b * total_changes); this is O(total_changes)). Matters most
    # for instance masks, where b is the per-slice instance count.
    row, col = change_indices[:, 0], change_indices[:, 1]
    boundaries = np.searchsorted(row, np.arange(b + 1))
    groups = np.split(col, boundaries[1:-1])

    # Additional metadata
    metadata = {}

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = groups[i]
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


def _encode_instance(mask: np.ndarray, **kwargs) -> list[dict]:
    """
    Encode instances masks into RLE format.

    This needs to be used for models like SAM, where instance masks overlap.
    Otherwise, binary encoding then connected components is likely faster and simpler.

    Each instance is encoded within its own bounding box rather than against
    the full slice. Previously runtime was dominated by np.argwhere, which
    scales with the area scanned, and dense masks can have 100+ instances per slice.
    Encoding every one of them against the full H*W frame dominates encode time
    even though any single instance typically occupies a small fraction of it.

    Assumes labels are small, densely-packed positive integers (as produced by
    connected-components/enumeration schemes, e.g. scipy.ndimage.label) --
    find_objects scales with the max label value, so sparse/huge label IDs
    would be inefficient here.
    """
    out = []
    # We need to loop over each slice
    for idx in range(mask.shape[0]):
        # Get the mask for this slice
        mask_slice = mask[idx]
        h, w = mask_slice.shape
        # Bounding box per instance label, in one call
        bboxes = ndimage.find_objects(mask_slice)
        encoded_masks = []
        for instance_id, bbox in enumerate(bboxes, start=1):
            if bbox is None:
                # This label isn't present in this slice
                continue
            y_slice, x_slice = bbox
            local_mask = mask_slice[bbox] == instance_id
            entry = _encode_binary(local_mask[np.newaxis, ...], idx=[instance_id])[0]
            entry["offset"] = [y_slice.start, x_slice.start]
            entry["full_size"] = [h, w]
            encoded_masks.append(entry)
        if not encoded_masks:
            # No instances in this slice
            encoded_masks = _encode_binary(
                np.zeros_like(mask_slice, dtype=bool)[np.newaxis, ...],
                idx=np.array([0], dtype=np.uint8),
            )
        # Store the encoded masks
        out.append(encoded_masks)
    return out


def decode(rle: list[dict], mask_type: str | None = None) -> tuple[np.ndarray, dict]:
    metadata = rle[-1]
    encoding = rle[:-1]

    # Try to get mask_type from metadata first, then parameter, then infer
    if mask_type is None:
        if "mask_type" in metadata.get("metadata", {}):
            mask_type = metadata["metadata"]["mask_type"]
        else:
            # Fall back to structure-based inference
            mask_type = check_rle_type(encoding)
            warnings.warn(
                f"Mask type not found in metadata, inferring as {mask_type}",
                stacklevel=2,
            )
    # TODO: Some basic checks for rle key validity
    if mask_type == "binary":
        res = _decode_binary(encoding)
    elif mask_type == "instance":
        # TODO: Some additional checks for keys for instance masks?
        res = _decode_instance(encoding)
    # NOTE: We squeeze here as any 2D inputs are unsqueezed to 3D for simplicity when encoding
    return res.squeeze(), metadata


def check_rle_type(rle: list[dict]) -> str:
    mask_type = "instance" if isinstance(rle[0], list) else "binary"
    return mask_type


def _decode_run_length(size: list[int], counts: list[int]) -> np.ndarray:
    """Decode one RLE entry's counts into its own (possibly bbox-local) boolean mask."""
    # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/utils/amg.py#L140
    h, w = size
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in counts:
        mask[idx : idx + count] = parity
        idx += count
        # This acts as a toggle
        parity ^= True
    if idx != h * w:
        # Check if counts are malformed and fail fast (again, only matters for masks outside AIoD)
        raise ValueError(
            f"Malformed RLE counts: expected total length {h * w}, got {idx}"
        )
    # Reshape and put in C order (encoded in Fortran order)
    return mask.reshape(w, h).transpose()


def _decode_binary(rle: list[dict]) -> np.ndarray:
    # Used directly for the top-level "binary" mask_type, where each entry is
    # always a full slice (never bbox-cropped), so no offset placement here.
    res = [_decode_run_length(entry["size"], entry["counts"]) for entry in rle]
    return np.stack(res, axis=0, dtype=bool)


def _decode_instance(rle) -> np.ndarray:
    # Container for unknown number of decoded mask slices
    out = []
    # rle_slice is a list of dictionaries for each instance
    for rle_slice in rle:
        # Full-frame size: new-format (bbox-cropped) entries carry it in
        # "full_size"; old-format entries (pre bounding-box encode) cover the
        # full frame directly in "size". Every entry in a slice shares one
        # full frame, so the first entry's is enough; _encode_instance always
        # produces at least one entry per slice, so rle_slice is never empty.
        first = rle_slice[0]
        h, w = first.get("full_size", first["size"])
        canvas = np.zeros((h, w), dtype=np.uint16)
        for entry in rle_slice:
            local_mask = _decode_run_length(entry["size"], entry["counts"])
            y0, x0 = entry.get("offset", (0, 0))
            lh, lw = entry["size"]
            # Write each instance directly into its own region of the shared
            # canvas instead of building a (K, H, W) stack to sum -- avoids
            # ever materializing full-frame arrays per instance. Genuinely
            # overlapping instances resolve as "last entry in the list wins"
            # here, rather than the old sum-based approach, which never
            # produced a correct value at overlaps anyway.
            canvas[y0 : y0 + lh, x0 : x0 + lw][local_mask] = entry["idx"]
        out.append(canvas)
    # Reconstruct the full mask array
    return np.stack(out)


def save_encoding(rle: list[dict], fpath: str | Path):
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


def load_encoding(fpath: str | Path) -> list[dict]:
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
