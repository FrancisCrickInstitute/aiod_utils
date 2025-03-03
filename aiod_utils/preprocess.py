from abc import abstractmethod
import json
from pathlib import Path
from typing import Optional, Union
import warnings

from cv2 import createCLAHE
import numpy as np
from skimage.measure import block_reduce
import skimage.io
import yaml


class Preprocess:
    # Display/readable name for the preprocessing function to use in UI
    name: str = None
    # Parameter dictionary for the underlying function
    # Should contain the parameters as keys
    # Under that then the default value and a pretty name for the UI
    params: dict = None
    # A tooltip for the UI
    tooltip: str = None
    # A flag to indicate if the function will change the image shape
    shape_change: bool = False

    def __init__(self, params: dict):
        # Check if the subclass has defined the required attributes
        # if self.name is None:
        #     raise ValueError(
        #         f"Preprocessing subclass ({self.__class__}) must have a name attribute!"
        #     )
        # if self.tooltip is None:
        #     raise ValueError(
        #         f"Preprocessing subclass ({self.__class__.name}) must have a tooltip attribute!"
        #     )
        # Construct the final parameters
        # Extract subclass' default params
        default_params = self.__class__.params
        self.kwarg_params = {k: v["default"] for k, v in default_params.items()}
        # Update with the provided parameters
        if len(params) > 0:
            for k in params:
                if k in self.kwarg_params:
                    self.kwarg_params[k] = params[k]
                else:
                    raise ValueError(
                        f"Invalid parameter for {self.__class__.name}: {k}"
                    )
        # Otherwise just extract the default parameters
        else:
            self.kwarg_params = {k: v["default"] for k, v in default_params}

    @abstractmethod
    def run(self, img):
        raise NotImplementedError

    # @abstractmethod
    # def check_image(self, img):
    #     """Check if the image is valid for the preprocessing function"""
    #     raise NotImplementedError

    def __str__(self) -> str:
        final_str = f"{self.name}-"
        for k, v in self.kwarg_params.items():
            final_str += f"{k}={v}-"
        return final_str[:-1]


class Downsample(Preprocess):
    name: str = "Downsample"

    shape_change: bool = True

    methods: dict = {
        "mean": np.mean,
        "median": np.median,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
    }

    params: dict = {
        "block_size": {
            "name": "Factor (D, H, W)",
            "default": (1, 2, 2),
            "tooltip": "Downsample factor for each dimension (D, H, W)",
        },
        "method": {
            "name": "Method",
            "default": "median",
            "values": list(methods.keys()),
            "tooltip": "Downsampling methods, i.e. how to aggregate the blocks",
        },
    }

    tooltip: str = "Downsample the image by a factor"

    def __init__(self, params: dict):
        if params["method"] not in self.methods:
            raise ValueError("Invalid method for downsampling!")
        super().__init__(params)

    def check_input(self, img_shape: tuple[int, ...]):
        # Check the block size is valid
        if len(self.kwarg_params["block_size"]) != len(img_shape):
            raise ValueError(
                f"Block size ({self.kwarg_params['block_size']}) must have the same length as the image shape ({img_shape})"
            )

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        self.check_input(input_shape)
        res = []
        for s, bs in zip(input_shape, self.kwarg_params["block_size"]):
            # Get the remainder as a result of padding
            out = int(np.ceil(s / bs))
            if s % bs == 0:
                res.append(out)
            else:
                res.append(out - (s % bs))
        return tuple(res)

    def run(self, img):
        orig_dtype = img.dtype
        self.check_input(img.shape)
        # Round the result to the nearest integer to avoid rounding down when casting back to original dtype
        res = np.round(
            block_reduce(
                img,
                block_size=tuple(self.kwarg_params["block_size"]),
                func=self.methods[self.kwarg_params["method"]],
            )
        ).astype(orig_dtype)
        # Store slice objects for each dim in turn
        slices = []
        changed = False
        # We care if padding was needed, so look at original image size
        for i, size in enumerate(img.shape):
            down_size = res.shape[i]
            pad_size = size % self.kwarg_params["block_size"][i]
            if pad_size == 0:
                # But we use the downsampled size for slicing
                # If divisable by block size, we can use the full size
                slices.append(slice(0, down_size))
            else:
                # Otherwise, remove the padded row(s)/col(s)/etc. from the downsampled image
                changed = True
                slices.append(slice(0, down_size - pad_size))
        if changed:
            warnings.warn(
                "Downsampling factor requires padding, so the image was cropped! Final result will have at least 1 pixel gap."
            )
        return res[tuple(slices)]


class CLAHE(Preprocess):
    """
    The skimage implementation requires a normalized clip_limit, which is unhelpful for ImageJ users.

    The OpenCV implementation lacks the bins argument of ImageJ, but as that's rarely used we can use it for now.
    Note that OpenCV's tileGridSize is seemingly a little different from ImageJ's block size, but it's close enough.

    For stacks and/or multi-channel images, CLAHE is applied to each channel/slice separately.
    """

    name: str = "CLAHE"
    params: dict = {
        "tileGridSize": {
            "name": "Tile/Block size",
            "default": (64, 64),
            "tooltip": "Size of the tile to equalize the histogram of",
        },
        "clipLimit": {
            "name": "Clip limit/Slope",
            "default": 5.0,
            "tooltip": "Clip limit for contrast, avoiding noise amplification",
        },
    }
    tooltip: str = "Contrast Limited Adaptive Histogram Equalization"

    def __init__(self, params: dict):
        super().__init__(params)

    def run(self, img):
        clahe_obj = createCLAHE(**self.kwarg_params)
        if img.ndim == 2:
            return clahe_obj.apply(img)
        # Multi-channel image or stack of single channel images
        elif img.ndim == 3:
            for i in range(img.shape[0]):
                img[i, :, :] = clahe_obj.apply(img[i, :, :])
        # Multi channel stack (CDHW)
        elif img.ndim == 4:
            # Iterate over the channels and stack
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i, j, :, :] = clahe_obj.apply(img[i, j, :, :])
        else:
            raise ValueError("Invalid image shape")
        return img


class Filter(Preprocess):
    name: str = "Filter"

    funcs: dict = {
        "mean": skimage.filters.rank.mean,
        "median": skimage.filters.rank.median,
    }

    filters: dict = {
        "square": skimage.morphology.square,
        "cube": skimage.morphology.cube,
        "disk": skimage.morphology.disk,
        "ball": skimage.morphology.ball,
    }

    params: dict = {
        "footprint": {
            "name": "Filter",
            "default": "disk",
            "values": list(filters.keys()),
            "tooltip": "Shape of the neighbourhood used for filtering",
        },
        "size": {
            "name": "Width/Radius",
            "default": 5,
            "tooltip": "Width of the square/cube or radius of the disk/ball used to define the neighbourhood",
        },
        "method": {
            "name": "Method",
            "default": "median",
            "values": list(funcs.keys()),
            "tooltip": "Filtering method to apply to the neighbourhood",
        },
    }

    tooltip: str = (
        "Apply a rank filter to the image. Note that 3D filters cannot be used on 2D images and vice versa."
    )

    def __init__(self, params: dict):
        if params["footprint"] not in self.filters:
            raise ValueError(
                f"Invalid neighbourhood/footprint option ({params['footprint']})! Must be one of {self.filters.keys()}"
            )
        if params["method"] not in self.funcs:
            raise ValueError(
                f"Invalid method ({params['method']}! Must be one of {self.funcs.keys()}"
            )
        super().__init__(params)

    def run(self, img):
        img = self.check_input(img)
        footprint = self.filters[self.kwarg_params["footprint"]](
            self.kwarg_params.pop("size")
        )
        return self.funcs[self.kwarg_params["method"]](img, footprint=footprint)

    def check_input(self, img):
        # skimage will throw an error if a 3D neighbourhood is used on a 2D image
        if img.ndim == 2:
            if self.kwarg_params["footprint"] in ["cube", "ball"]:
                raise ValueError(
                    "A 3D filter (cube/ball) cannot be used on a 2D image!"
                )
        # skimage will throw an error if a 2D neighbourhood is used on a 3D image
        elif img.ndim == 3:
            if self.kwarg_params["footprint"] in ["square", "disk"]:
                raise ValueError(
                    "A 2D filter (square/disk) cannot be used on a 3D image!"
                )
        elif img.ndim > 3:
            raise ValueError("Filter only works with 2D or 3D images!")
        return img


def run_method(
    img: np.ndarray, method: Optional[str] = None, params: Optional[dict] = None
):
    # Get the selected preprocess class
    preprocess_cls = {cls.name: cls for cls in Preprocess.__subclasses__()}[method]
    # Create instance with args
    cls = preprocess_cls(params=params)
    # Run the preprocess
    return cls.run(img)


def check_method(method, params):
    # Check the method name maps to a class
    if method not in [cls.name for cls in Preprocess.__subclasses__()]:
        raise ValueError(
            f"Invalid preprocess method: {method}. Must be one of {(i.name for i in Preprocess.__subclasses__())}"
        )
    # Check the parameters are valid
    try:
        {cls.name: cls for cls in Preprocess.__subclasses__()}[method](params=params)
    except Exception as e:
        raise ValueError(f"Invalid parameters for {method} ({params}): {e}")


def load_methods(methods: Union[list[dict], str, Path], parse: bool = True):
    if isinstance(methods, (str, Path)):
        methods = Path(methods)
        # Handle JSON
        if methods.suffix == ".json":
            with open(methods, "r") as f:
                methods = json.load(f)
        # Handle YAML
        elif methods.suffix in [".yaml", ".yml"]:
            with open(methods, "r") as f:
                methods = yaml.safe_load(f)
    if parse:
        return parse_methods(methods)
    else:
        return methods


def parse_methods(methods: Optional[list[dict]]):
    if methods is None:
        return []
    for method in methods:
        check_method(method["name"], method["params"])
    return methods


def run_preprocess(img: np.ndarray, methods: Optional[Union[list[dict], str, Path]]):
    # Load and check all methods are valid
    methods = load_methods(methods, parse=True)
    # If no method is specified, return the original image
    if len(methods) == 0:
        return img
    # Run the methods in order
    for method in methods:
        img = run_method(img, method["name"], method["params"])
    return img


# TODO: Rename to 'all', as this implies selected
def get_preprocess_methods():
    # Return a dictionary of the available preprocess subclasses sorted by name
    return dict(
        sorted(
            {
                cls.name: {"object": cls, "params": cls.params}
                for cls in Preprocess.__subclasses__()
            }.items()
        )
    )


def get_preprocess_params(methods: Optional[Union[list[dict], str, Path]]) -> str:
    # Load and check all methods are valid
    methods = load_methods(methods, parse=True)
    # If no method is specified, return the original image
    if len(methods) == 0:
        return ""
    # Run the methods in order
    res = []
    for method_dict in methods:
        method, params = method_dict["name"], method_dict["params"]
        # Get the selected preprocess class
        preprocess_cls = {cls.name: cls for cls in Preprocess.__subclasses__()}[method]
        # Create instance with args
        cls = preprocess_cls(params=params)
        res.append(str(cls))
    return "_".join(res)


def get_downsample_factor(
    methods: Optional[Union[list[dict], str, Path]]
) -> Optional[tuple[int, ...]]:
    # Load and check all methods are valid
    methods = load_methods(methods, parse=True)
    factor = None
    for d in methods:
        if d["name"] == "Downsample":
            factor = d["params"]["block_size"]
            break
    return factor


def get_output_shape(options, input_shape: tuple[int, ...]):
    # Load and check all methods are valid
    methods = load_methods(options, parse=True)
    # If no method is specified, return the input shape
    if len(methods) == 0:
        return input_shape
    output_shape = input_shape
    for method_dict in methods:
        method, params = method_dict["name"], method_dict["params"]
        # Get the selected preprocess class
        preprocess_cls = {cls.name: cls for cls in Preprocess.__subclasses__()}[method]
        # Only update the shape if the method changes it
        if preprocess_cls.shape_change:
            cls = preprocess_cls(params=params)
            output_shape = cls.get_output_shape(output_shape)
    return output_shape
