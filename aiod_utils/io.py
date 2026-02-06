from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
import warnings

from bioio import BioImage
from bioio_base.reader import Reader
import numpy as np
import pandas as pd
from skimage.io import imread


def load_image(
    fpath: str | Path,
    return_array: bool = False,
    return_dask: bool = False,
    dim_order="CZYX",
    sense_check: bool = True,
    ensure_channel_lower_depth: bool = True,
    reader: type[Reader] | None = None,
) -> BioImage:
    assert Path(fpath).exists(), f"File {fpath} does not exist!"
    fpath = Path(fpath)
    # If returning actual data, we need to know the dimension order
    if dim_order is None and any([return_array, return_dask]):
        raise ValueError(
            "If you want to return an array or a dask array, you must specify the dimension order!"
        )
    # You cannot return both an array and a dask array
    if return_array and return_dask:
        raise ValueError("You cannot return both an array and a dask array!")
    # Check the dim_order, and remap obvious aliases
    dim_order = dim_order.upper()
    dim_order = dim_order.translate(str.maketrans("DHW", "ZYX"))
    # NOTE: Had some issues with jpg/png, so use skimage for those
    if fpath.suffix in [".jpg", ".jpeg", ".png"]:
        warnings.warn(
            f"Using skimage (not bioio) to load {fpath.name} due to issues with bioio and jpg/png. All bioio arguments are ignored."
        )
        return imread(fpath)
    # Load the image with the requested reader
    # Default reader is None, and bioio determines which to use (most recently installed for that extension)
    img = BioImage(fpath, reader=reader)
    # Do some basic checks to flag potential issues
    if sense_check:
        if img.dims.Z > 1:
            if img.dims.C > img.dims.Z:
                # Manually swap the channel and depth dims, for whatever order is given, if requested
                if ensure_channel_lower_depth:
                    dim_order = dim_order.translate(str.maketrans("CZ", "ZC"))
                    # This has no effect if not returning an array
                    # TODO: Look into manipulating BioIO dims to force change when loading data later
                    if not any([return_array, return_dask]):
                        warnings.warn(
                            "If ensure_channel_lower_depth=True, you should also return an array or a dask array otherwise nothing changes!"
                        )
                else:
                    warnings.warn(
                        f"Image {fpath.name} has more channels ({img.dims.C}) than slices ({img.dims.Z})! Is that right? Can be fixed with ensure_channel_lower_depth=True."
                    )
            if img.dims.Y < img.dims.Z:
                warnings.warn(
                    f"Image {fpath.name} has more slices ({img.dims.Z}) than height ({img.dims.Y})! Is that right?"
                )
            if img.dims.X < img.dims.Z:
                warnings.warn(
                    f"Image {fpath.name} has more slices ({img.dims.Z}) than width ({img.dims.X})! Is that right?"
                )
    # Return in the desired format
    # NOTE: Could add idxs here, but as they load into memory there's no advantage
    # So we leave that for Segment-Flow to do as it's almost exclusively done there
    if return_array:
        return img.get_image_data(dimension_order_out=dim_order)
    elif return_dask:
        return img.get_image_dask_data(dimension_order_out=dim_order)
    else:
        return img


def image_paths_to_csv(
    image_paths: Sequence[str | Path] | str | Path,
    output_csv_path: str | Path,
    dimensions: Sequence[dict[str, int]] | dict[str, int] | None = None,
    overwrite: bool = False,
    **kwargs
):
    '''
    Write image shape details to a csv file, given an input image path or list of paths.
    Optionally provide shape details (per image path) to be written in the form of dimensions dict with keys from STCZYX:
        dimensions = {
            'X':...,
            'Z':...,
        }
    If shape info are not provided or are incomplete, will attempt to read from the image metadata if available.
    Will raise FileExistsError if overwrite=False and output_csv_path exists.
    Any additional kwargs will be forwarded to pandas.DataFram.to_csv()
    '''

    if not overwrite and Path(output_csv_path).exists():
        raise FileExistsError(f"Output csv file {output_csv_path} already exists and overwrite is set to False.")
    
    output = defaultdict(list)
    
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    if dimensions:
        if isinstance(dimensions, dict):
            dimensions = [dimensions]
        if len(dimensions) != len(image_paths):
            raise ValueError("If providing dimensions, must provide one dimensions dict per image path.")
    else:
        # Fetch dimensions for each image from metadata
        raise NotImplementedError("Fetching dimensions from image metadata not yet implemented.")
        
    for path, shape in zip(image_paths, dimensions):
        output["img_path"].append(str(path))
        try:
            output["num_slices"].append(shape['Z'])
            output["height"].append(shape.get('Y') or shape['H']) # raises KeyError
            output["width"].append(shape.get('X') or shape['W']) # raises KeyError
            output["channels"].append(shape.get('C', 1))
        except KeyError as e:
            # NOTE: this message will give keyerror for H or W, without hinting to use Y and X instead
            raise ValueError(f"Dimensions dict for image {path} is missing required key: {e}")
    df = pd.DataFrame(output)
    df.to_csv(output_csv_path, **kwargs)


def extract_idxs_from_fname(
    fname: str, downsample_factor: Sequence[int] | None = None
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


def check_dtype(arr: np.ndarray, max_val: int | None = None):
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


def reduce_dtype(arr: np.ndarray, max_val: int | None = None):
    # Get the lowest bit dtype for the array
    best_dtype = check_dtype(arr, max_val)
    # If the current dtype is already the best, return the array
    if arr.dtype == best_dtype:
        return arr
    # Otherwise convert it
    else:
        return arr.astype(best_dtype, copy=False)

