from pathlib import Path
from typing import Union, Optional, Type
import warnings

from bioio import BioImage
from bioio_base.reader import Reader
import numpy as np
import dask.array as da


def _guess_reader(fpath: Union[str, Path]):
    ext = Path(fpath).suffix.lower()
    try:
        if ext in [".tif", ".tiff"]:
            # TODO: add explicit .ome.tiff support
            from bioio_tifffile import Reader as TiffReader
            return TiffReader
        elif ext in [".zarr"]:
            from bioio_ome_zarr import Reader as ZarrReader
            return ZarrReader
        elif ext in [".jpg", ".jpeg", ".png"]:
            from bioio_imageio import Reader as ImageIOReader
            return ImageIOReader
        elif ext in [".nd2"]:
            from bioio_nd2 import Reader as ND2Reader
            return ND2Reader
    except ModuleNotFoundError as e:
        warnings.warn(
            f"Recommended reader plugin {e.name} for file type {ext} not installed"
        )
    return None


def guess_rgba(img: BioImage):
    # https://github.com/bioio-devs/bioio/issues/174#issuecomment-3843003521
    return 'S' in img.dims.order


def load_image_data(
    fpath: Union[str, Path], dim_order: str = "CZYX", as_dask: bool = True, **kwargs
) -> np.ndarray | da.Array:
    """
    Returns data array without any associated metadata.
    Replaces legacy flagged mode of load_image:
        load_image(...,) => load_image(...,)
        load_image(..., return_array=True) => load_image_data(...)
        load_image(..., return_dask=True) => load_image_data(..., as_dask=True)
    """
    img = load_image(fpath, **kwargs)
    # Check the dim_order, and remap obvious aliases
    dim_order = dim_order.upper().translate(str.maketrans("DHW", "ZYX"))
    #TODO: S -> C ("samples" means RGB(A) colour channel)
    if "C" in dim_order and "S" not in dim_order and guess_rgba(img):
        if getattr(img.dims, "C", 1) > 1:
            raise NotImplementedError("Multi-channel RGB(A) images not supported")
        dim_order = dim_order.replace("C", "S")
    return (
        img.get_image_dask_data(dimension_order_out=dim_order)
        if as_dask
        else img.get_image_data(dimension_order_out=dim_order)
    )


def load_image(
    fpath: Union[str, Path],
    reader: Optional[Type[Reader]] = None,
) -> BioImage:
    # Load the image with the requested reader
    # If no reader provided, guess an appropriate plugin or fall back on bioio default plugin order
    fpath = Path(fpath)
    return BioImage(fpath, reader=reader or _guess_reader(fpath))


def extract_idxs_from_fname(
    fname: str, downsample_factor: Optional[list[int, ...]] = None
):
    # Extract the indices from the filename
    idx_ranges = Path(fname).stem.split("_")[-3:]
    start_x, end_x = map(int, idx_ranges[0].split("x")[1].split("-"))
    start_y, end_y = map(int, idx_ranges[1].split("y")[1].split("-"))
    start_z, end_z = map(int, idx_ranges[2].split("z")[1].split("-"))
    # Apply downsampling to indices if provided
    if downsample_factor is not None:
        if len(downsample_factor) == 2:
            down_y, down_x = downsample_factor
            down_z = 1
        else:
            down_z, down_y, down_x = downsample_factor
        start_x, end_x = round_idxs(start_x, end_x, down_x)
        start_y, end_y = round_idxs(start_y, end_y, down_y)
        start_z, end_z = round_idxs(start_z, end_z, down_z)
    return start_x, end_x, start_y, end_y, start_z, end_z


def round_idxs(start: int, end: int, downsample_factor: int):
    """
    When splitting and downsampling, we end up with indivisible block sizes.
    We handle this by removing whatever is getting padded in block_reduce
    We need to convert idxs in fnames to downsampled idxs, which may not divide cleanly.
    If we round down the start we increase size, so need to round different according to whether start or end is indivisable.
    """
    if (end - start) % downsample_factor != 0:
        start = int(np.ceil(start / downsample_factor))
        end = int(np.floor(end / downsample_factor))
    else:
        start = int(np.floor(start / downsample_factor))
        end = int(np.floor(end / downsample_factor))
    return start, end


def check_dtype(arr: np.ndarray, max_val: Optional[int] = None):
    # Get the max value in the array if not provided
    if max_val is None:
        max_val = arr.max()
    # Get the appropriate dtype from the max value
    if max_val <= np.iinfo(np.uint8).max:
        best_dtype = np.uint8
    elif max_val <= np.iinfo(np.uint16).max:
        best_dtype = np.uint16
    # Surely it doesn't need more than 32 bits...
    else:
        best_dtype = np.uint32
    return best_dtype


def reduce_dtype(arr: np.ndarray, max_val: Optional[int] = None):
    # Get the lowest bit dtype for the array
    best_dtype = check_dtype(arr, max_val)
    # If the current dtype is already the best, return the array
    if arr.dtype == best_dtype:
        return arr
    # Otherwise convert it
    else:
        return arr.astype(best_dtype, copy=False)

