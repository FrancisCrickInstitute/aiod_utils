from pathlib import Path
from typing import Union, Optional, Type
import warnings

from bioio import BioImage
from bioio_base.reader import Reader


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
    # If returning actual data, we need to know the dimension order
    if dim_order is None and any([return_array, return_dask]):
        raise ValueError(
            "If you want to return an array or a dask array, you must specify the dimension order!"
        )
    # You cannot return both an array and a dask array
    if return_array and return_dask:
        raise ValueError("You cannot return both an array and a dask array!")
    # Check the dim_order, and remap obvious aliases
    dim_order = dim_order.translate(str.maketrans("DHW", "ZYX"))
    # Load the image with the requested reader
    # Default reader is None, and bioio determines which to use
    # NOTE: bioio uses the most recently installed reader that matches the extension
    img = BioImage(fpath, reader=reader)
    # Do some basic checks to flag potential issues
    if sense_check:
        if img.dims.Z > 1:
            if img.dims.C > img.dims.Z:
                if ensure_channel_lower_depth:
                    dim_order = "ZCYX"
                    # This has no effect if not returning an array
                    if not any([return_array, return_dask]):
                        raise ValueError(
                            "If ensure_channel_lower_depth=True, you should also return an array or a dask array!"
                        )
                else:
                    warnings.warn(
                        f"Image {fpath} has more channels ({img.dims.C}) than slices ({img.dims.Z})! Is that right? Can be fixed with ensure_channel_lower_depth=True."
                    )
            if img.dims.Y > img.dims.Z:
                warnings.warn(
                    f"Image {fpath} has more height ({img.dims.Y}) than slices ({img.dims.Z})! Is that right?"
                )
            if img.dims.X > img.dims.Z:
                warnings.warn(
                    f"Image {fpath} has more width ({img.dims.X}) than slices ({img.dims.Z})! Is that right?"
                )
    # Return in the desired format
    # NOTE: Could add idxs here, but as they load into memory there's no advantage
    if return_array:
        return img.get_image_data(dimension_order_out=dim_order)
    elif return_dask:
        return img.get_image_dask_data(dimension_order_out=dim_order)
    else:
        return img
