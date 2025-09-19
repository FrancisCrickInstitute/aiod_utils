from pathlib import Path
from typing import Union, Optional, Type
import warnings

from bioio import BioImage
from bioio_base.reader import Reader
from bioio_base.writer import Writer
import numpy as np
import dask.array as da
from skimage.io import imread, imsave


def load_image(
    fpath: Union[str, Path],
    return_array: bool = False,
    return_dask: bool = False,
    dim_order="CZYX",
    sense_check: bool = True,
    ensure_channel_lower_depth: bool = True,
    reader: Optional[Type[Reader]] = None,
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

def resize_dim(img):
    original_shape = img.shape  # Store the original shape
    # Acceptable shapes: (M, N), (M, N, 3), (M, N, 4)
    if len(original_shape) == 2 or (len(original_shape) == 3 and original_shape[2] in [3, 4]):
        return img
    # Squeeze singleton dimensions
    squeezed = np.squeeze(img)
    shape = squeezed.shape
    if len(shape) == 2 or (len(shape) == 3 and shape[2] in [3, 4]):
        warnings.warn(f"Resized image from {original_shape} to {shape} for saving.")
        return squeezed
    # If still too many dimensions, select the first slice along extra axes
    while len(shape) > 3:
        squeezed = squeezed[0]
        shape = squeezed.shape
    if len(shape) == 2 or (len(shape) == 3 and shape[2] in [3, 4]):
        warnings.warn(f"Selected first slice, resized image from {original_shape} to {shape} for saving.")
        return squeezed
    raise ValueError(f"Cannot convert image to a supported shape for imsave. Got shape {original_shape}")

def save_image(
    img,
    save_dir: Union[str, Path],
    save_name: str,
    save_format: str,
    save_multi: Optional[bool] = False, # Save each image separate or in 1 image
    dtype: Optional[np.dtype] = None,
    metadata: Optional[dict] = None, # not supported with skimage imsave
    writer: Optional[Type[Writer]] = None, # for future implementation
    **kwargs,
):
    """
    Universal save image function numpy array or BioImage
    - JPG/PNG formats only support 2D images or 2D + 3/4 channels. Extra dimensions are squeezed or sliced automatically.
    - The `dtype` parameter converts the array before saving. jpg and png don't support all dtypes.
    - Currently metadata isn't preserved (needs bioio reader implementation with ome.tiff)
    """
    save_dir = Path(save_dir)
    assert Path(save_dir).exists(), f"path:{save_dir} doesn't exist"
    if not isinstance(img, np.ndarray):
        if hasattr(img, "get_image_data"):
            img = img.get_image_data()
        elif isinstance(img, da.Array):
            img = img.compute() #skimage imsave doesn't support dask arrays
        else:
            raise TypeError("Unsupported image type: Must be numpy array or have get_image_data() method")
    if dtype is not None:
        img = reduce_dtype(img, dtype=dtype)
    if save_format in ['jpg', 'jpeg', 'png']:
        img = resize_dim(img)
        try:
            save_path = save_dir / f"{save_name}.{save_format}"
            imsave(save_path, img)
        except Exception as e:
            raise IOError(f"Failed to save image '{save_name}.{save_format}': {e}")
    # elif save_format in ['ome.tiff', 'ome.tif']:
    #     bio_img = BioImage(img, metadata=metadata)
    #     bio_img.save(f"{save_dir}/{save_name}.{save_format}")
    else:
        if save_multi:
            if len(img.shape) == 5: 
                for t in range(img.shape[0]):
                    for c in range(img.shape[1]):
                        for z in range(img.shape[2]):
                            slice_img = img[t, c, z]
                            slice_name = f"{save_name}_T{t}C{c}Z{z}.{save_format}"
                            try:
                                save_path = save_dir / slice_name
                                imsave(save_path, slice_img)
                            except Exception as e:
                                raise IOError(f"Failed to save image '{slice_name}': {e}")
            else:
                raise ValueError(f"Cannot save image: unsupported shape {img.shape}. Expected 5D (T, C, Z, Y, X) for multi-slice saving.")
        else:
            try:
                save_path = save_dir / f"{save_name}.{save_format}"
                imsave(save_path, img)
            except Exception as e:
                raise IOError(f"Failed to save image '{save_name}.{save_format}': {e}")




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


def reduce_dtype(arr: np.ndarray, max_val: Optional[int] = None, dtype: Optional[np.dtype] = None):
    # If dtype is provided, use it directly
    if dtype is not None:
        best_dtype = dtype
    else:
        best_dtype = check_dtype(arr, max_val)
    # If the current dtype is already the best, return the array
    if arr.dtype == best_dtype:
        return arr
    # Otherwise convert it
    else:
        return arr.astype(best_dtype, copy=False)