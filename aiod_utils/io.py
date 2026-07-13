import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from bioio import BioImage, writers
from bioio_base.exceptions import InvalidDimensionOrderingError
from bioio_base.reader import Reader

PathLike = str | Path
ImageLike = BioImage | np.ndarray | da.Array


def _guess_reader(fpath: PathLike) -> type[Reader] | None:
    ext = "".join(Path(fpath).suffixes).lower()
    try:
        if ext.endswith((".ome.tiff", ".ome.tif")):
            from bioio_ome_tiff import Reader as OMETiffReader

            return OMETiffReader
        elif ext.endswith((".tif", ".tiff")):
            from bioio_tifffile import Reader as TiffReader

            return TiffReader
        elif ext.endswith((".zarr", ".ome.zarr")):
            from bioio_ome_zarr import Reader as ZarrReader

            return ZarrReader
        elif ext.endswith((".jpg", ".jpeg", ".png")):
            from bioio_imageio import Reader as ImageIOReader

            return ImageIOReader
        elif ext.endswith((".nd2",)):
            from bioio_nd2 import Reader as ND2Reader

            return ND2Reader
    except ModuleNotFoundError as e:
        warnings.warn(
            f"Recommended reader plugin {e.name} for file extension {ext} not installed",
            stacklevel=2,
        )
    return None


def guess_rgba(img: BioImage):
    # https://github.com/bioio-devs/bioio/issues/174#issuecomment-3843003521
    return "S" in img.dims.order


def load_image_data(
    image: PathLike | BioImage,
    dim_order: str = "CZYX",
    as_dask: bool = False,
    rgb_as_channels=True,
    expand_dims=True,
    **kwargs,
) -> np.ndarray | da.Array:
    """
    Returns data array without any associated metadata.
    Replaces legacy flagged mode of load_image:
        load_image(...,) => load_image(...,)
        load_image(..., return_array=True) => load_image_data(...)
        load_image(..., return_dask=True) => load_image_data(..., as_dask=True)

    Inputs
    ======

    image: file path or BioImage object

    Note
    ====

    In Bioio, RGB images by default store the RGB dimension as in samples 'S', separate from channels 'C' (see bioio-devs/bioio#174). If `rgb_as_channels` is True, and if 'C' is requested in the output `dim_order`, the 'S' dimension will be remapped to 'C'.
    """
    if isinstance(image, (str, Path)):
        image = load_image(image, **kwargs)
    elif len(kwargs):
        warnings.warn(
            f"load_image_data() received unexpected kwargs {kwargs}, which will be ignored",
            stacklevel=2,
        )
    # Check the dim_order, and remap obvious aliases
    dim_order = dim_order.upper().translate(str.maketrans("DHW", "ZYX"))
    if (
        rgb_as_channels
        and "C" in dim_order
        and "S" not in dim_order
        and guess_rgba(image)
    ):
        if getattr(image.dims, "C", 1) > 1:
            raise NotImplementedError("Multi-channel RGB(A) images not supported")
        dim_order = dim_order.replace("C", "S")
    # Keep only actual dims if singleton expansion disabled
    if not expand_dims:
        dim_order = "".join(
            d for d in dim_order if d in image.standard_metadata.dimensions_present
        )
    return (
        image.get_image_dask_data(dimension_order_out=dim_order)
        if as_dask
        else image.get_image_data(dimension_order_out=dim_order)
    )


def load_image(
    fpath: PathLike,
    reader: type[Reader] | None = None,
) -> BioImage:
    # Load the image with the requested reader
    # If no reader provided, guess an appropriate plugin or fall back on bioio default plugin order
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist.")
    return BioImage(fpath, reader=reader or _guess_reader(fpath))


def save_image(
    data: ImageLike,
    fpath: PathLike,
    dim_order: str = "CZYX",
):
    ext = "".join(Path(fpath).suffixes).lower()
    if ext.endswith((".ome.tiff", ".ome.tif", ".tif", ".tiff")):
        try:
            _save_image_ome_tiff(data, fpath, dim_order)
        except AttributeError as exc:
            # Fall back on tifffile for missing writer
            from tifffile import imwrite

            data = (
                load_image_data(
                    data,
                    dim_order=dim_order,
                    as_dask=False,
                )
                if isinstance(data, (BioImage, Path, str))
                else data
            )
            # TODO AIOD-315: can't deal with this scenario until load_image_data returns dim string
            if len(dim_order) != len(data.shape):
                raise NotImplementedError(
                    "Cannot use tifffile to save image with unspecified dimensions"
                ) from exc
            imwrite(fpath, data, metadata={"axes": dim_order})
    elif ext.endswith((".zarr", ".ome.zarr")):
        try:
            _save_image_ome_zarr(data, fpath, dim_order)
        except AttributeError as exc:
            raise NotImplementedError(
                "Cannot save to zarr without bioio-zarr installed."
            ) from exc
    else:
        raise ValueError(f"Unsupported extension: {ext}")


def _save_image_ome_zarr(data: ImageLike, fpath: PathLike, dim_order="CZYX"):
    if not hasattr(writers, "OMEZarrWriter"):
        raise AttributeError("OMEZarrWriter")
    if isinstance(data, BioImage):
        data = load_image_data(data, dim_order=dim_order)
    # Ensure axes_names length matches data ndim; take trailing axes if dim_order is longer
    if len(dim_order) > data.ndim:
        # NOTE: revisit this for AIOD-315
        dim_order = dim_order[-data.ndim :]
    elif len(dim_order) < data.ndim:
        raise InvalidDimensionOrderingError(
            f"dim_order '{dim_order}' has fewer dims than data shape {data.shape}"
        )
    writers.OMEZarrWriter(
        store=str(fpath),
        level_shapes=data.shape,
        dtype=data.dtype,
        axes_names=[a.lower() for a in dim_order],
    ).write_full_volume(data)


def _save_image_ome_tiff(data: ImageLike, fpath: PathLike, dim_order="CZYX"):
    if not hasattr(writers, "OmeTiffWriter"):
        raise AttributeError("OmeTiffWriter")
    if isinstance(data, BioImage):
        data.save(fpath, dim_order=dim_order)
    else:
        writers.OmeTiffWriter.save(data, fpath, dim_order=dim_order)


def image_paths_to_csv(
    image_paths: Sequence[PathLike] | PathLike,
    output_csv_path: PathLike,
    dimensions: Sequence[dict[str, int]] | dict[str, int] | None = None,
    dtypes: Sequence[str | np.dtype] | str | np.dtype | None = None,
    overwrite: bool = False,
    **kwargs,
):
    """
    Write image shape details to a csv file, given an input image path or list of paths.
    Optionally provide shape details (per image path) to be written in the form of dimensions dict with keys from STCZYX:
        dimensions = {
            'X':...,
            'Z':...,
        }
    If shape info are not provided or are incomplete, will attempt to read from the image metadata if available.
    Will raise FileExistsError if overwrite=False and output_csv_path exists.
    Any additional kwargs will be forwarded to pandas.DataFram.to_csv()
    """

    if not overwrite and Path(output_csv_path).exists():
        raise FileExistsError(
            f"Output csv file {output_csv_path} already exists and overwrite is set to False."
        )

    output = defaultdict(list)

    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    if dimensions:
        if isinstance(dimensions, dict):
            dimensions = [dimensions]
        if len(dimensions) != len(image_paths):
            raise ValueError(
                "If providing dimensions, must provide one dimensions dict per image path."
            )
    else:
        # Fetch dimensions for each image from metadata
        # TODO: When this is implemented, might as well enable fetching dtype while we're at it
        raise NotImplementedError(
            "Fetching dimensions from image metadata not yet implemented."
        )
    if dtypes is not None:
        if isinstance(dtypes, (str, np.dtype)):
            dtypes = [dtypes]
        if len(dtypes) != len(image_paths):
            raise ValueError(
                "If providing dtypes, must provide one dtype per image path."
            )
    dt_iter = dtypes if dtypes is not None else [None] * len(image_paths)
    for path, shape, dt in zip(image_paths, dimensions, dt_iter, strict=True):
        output["img_path"].append(str(path))
        try:
            output["num_slices"].append(shape.get("Z", 1))
            output["height"].append(shape.get("Y") or shape["H"])  # raises KeyError
            output["width"].append(shape.get("X") or shape["W"])  # raises KeyError
            output["channels"].append(shape.get("C", 1))
        except KeyError as e:
            # NOTE: this message will give keyerror for H or W, without hinting to use Y and X instead
            raise ValueError(
                f"Dimensions dict for image {path} is missing required key: {e}"
            ) from e
        if dt is not None:
            output["dtype"].append(np.dtype(dt).name)
        else:
            output["dtype"].append(None)
    df = pd.DataFrame(output)
    df.to_csv(output_csv_path, **kwargs)


def extract_idxs_from_fname(fname: str, downsample_factor: Sequence[int] | None = None):
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
