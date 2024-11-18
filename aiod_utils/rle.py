from pathlib import Path
import pickle
from typing import Union, Optional, Any
import warnings

import numpy as np
import torch

EXTENSIONS = [".pkl", ".pickle", ".rle"]


def encode(
    mask: Union[np.ndarray, torch.Tensor],
    mask_type: Optional[str] = None,
    metadata: dict[str, Any] = {},
) -> list[dict]:
    assert isinstance(
        mask, (np.ndarray, torch.Tensor)
    ), f"mask must be a numpy array or torch tensor, not {type(mask)}"
    # Convert to torch tensor
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    # Try to infer the mask type if not provided
    if mask_type is None:
        mask_type = check_mask_type(mask)
        warnings.warn(f"Mask type not provided, inferring as {mask_type}")
    # Give a batch dimension if it's not there
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask_type == "binary":
        # Ensure mask is boolean (rather than 2 unique values, it's faster)
        mask = mask.bool()
        res = _encode_binary(mask, **metadata)
    elif mask_type == "instance":
        mask = mask.long()
        res = _encode_instance(mask, **metadata)
    # Insert metadata
    res.append({"metadata": metadata})
    return res


def check_mask_type(mask: torch.Tensor) -> str:
    # Boolean masks are binary
    if mask.dtype == torch.bool:
        mask_type = "binary"
    # Masks with only 2 unique values are binary
    elif mask.unique().shape[0] <= 2:
        mask_type = "binary"
    # Otherwise, it's an instance mask
    else:
        mask_type = "instance"
    return mask_type


def _encode_binary(mask, **kwargs) -> list[dict]:
    # https://github.com/facebookresearch/sam2/blob/c2ec8e14a185632b0a5d8b161928ceb50197eddc/sam2/utils/amg.py#L109
    # B here is some kind of batch
    # For binarised instance segmentation, it's just a single mask
    # Otherwise it's a slice
    b, h, w = mask.shape
    mask = mask.permute(0, 2, 1).flatten(1)

    # Compute change indices
    # Essentially, XOR the mask with itself shifted by 1, identifying contiguous regions
    diff = mask[:, 1:] ^ mask[:, :-1]
    # Then find all the indices where we have a change
    change_indices = diff.nonzero()

    # Additional metadata
    metadata = {}

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        # Calculate the run length
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        # Start empty if first pixel is background, otherwise start with 0
        counts = [] if mask[i, 0] == 0 else [0]
        # Detach tensor for easier saving
        counts.extend(btw_idxs.detach().cpu().tolist())
        # Store the size and RLE/counts and any additional metadata
        if "idx" in kwargs:
            metadata = {"idx": int(kwargs["idx"][i])}
        out.append({"size": [h, w], "counts": counts, **metadata})
    return out


def _encode_instance(mask: torch.Tensor, **kwargs) -> list[dict]:
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
        instances = torch.unique(mask_slice)
        # Remove the background
        instances = instances[instances != 0]
        # Convert into a batch of binarised masks for each instance
        mask_batch = mask_slice.unsqueeze(0) == instances.unique().view(-1, 1, 1)
        # Encode the binary masks
        # Add the instance index to the metadata for later decoding
        encoded_masks = _encode_binary(mask_batch, idx=instances, **kwargs)
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
    return res, metadata


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
    return np.stack(res)


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
        instance_indices = np.array([r["idx"] for r in rle_slice], dtype=np.uint32)
        # Multiply the binary masks by the instance indices and sum to flatten
        decoded_slice = (decoded_slice * instance_indices[:, None, None]).sum(axis=0)
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
