from abc import abstractmethod
import json
from pathlib import Path
from typing import Optional, Union

from cv2 import createCLAHE
import matplotlib.pyplot as plt
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
        return tuple(
            [
                int(np.ceil(s / bs))
                for s, bs in zip(input_shape, self.kwarg_params["block_size"])
            ]
        )

    def run(self, img):
        orig_dtype = img.dtype
        self.check_input(img.shape)
        # Round the result to the nearest integer to avoid rounding down when casting back to original dtype
        return np.round(
            block_reduce(
                img,
                block_size=tuple(self.kwarg_params["block_size"]),
                func=self.methods[self.kwarg_params["method"]],
            )
        ).astype(orig_dtype)


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


def load_methods(methods: Union[list[dict], str, Path]):
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
    return methods


def parse_methods(methods: Optional[list[dict]]):
    if methods is None:
        return []
    for method in methods:
        check_method(method["name"], method["params"])
    return methods


def run_preprocess(img: np.ndarray, methods: Optional[Union[list[dict], str, Path]]):
    methods = load_methods(methods)
    # Check all methods are valid
    methods = parse_methods(methods)
    # If no method is specified, return the original image
    if len(methods) == 0:
        return img
    # Run the methods in order
    for method in methods:
        img = run_method(img, method["name"], method["params"])
    return img


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
    methods = load_methods(methods)
    # Check all methods are valid
    methods = parse_methods(methods)
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


def get_output_shape(options, input_shape: tuple[int, ...]):
    methods = load_methods(options)
    methods = parse_methods(methods)
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


if __name__ == "__main__":
    # Raw/original image to use as a test
    IMG_PATH = (
        "/Users/shandc/Documents/data/clahe/Tif_stack_Lorian_8bits-z600-700-crop.tif"
    )

    preprocess = [
        # {"name": "CLAHE", "params": {"tileGridSize": [64, 64], "clipLimit": 5}},
        # {"name": "Downsample", "params": {"method": "median", "block_size": [2, 2]}},
        {
            "name": "Filter",
            "params": {"footprint": "ball", "size": 5, "method": "median"},
        },
    ]

    img = skimage.io.imread(IMG_PATH)[:5]
    new_img = run_preprocess(img, preprocess)

    if img.ndim > 2:
        img = img[0]
        new_img = new_img[0]

    fig, ax = plt.subplots(1, 2)
    ax = ax.flatten()
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original")
    ax[1].imshow(new_img, cmap="gray")
    ax[1].set_title("Processed")
    fig.tight_layout()
    plt.show()

    # Externally transformed image to compare against
    # METHOD = "clahe"
    # METHOD = "downsample"
    # IMG_TRANSFORM_PATH = {
    #     "clahe": "/Users/shandc/Documents/data/clahe/Tif_stack_Lorian_8bits-z600-700-crop_CLAHE3.tif",
    #     "downsample": "/Users/shandc/Documents/data/clahe/Tif_stack_Lorian_8bits-z600-700-crop_binner-sum-cs.tif",
    # }[METHOD]

    # orig = skimage.io.imread(IMG_PATH)  # [0]
    # transformed = skimage.io.imread(IMG_TRANSFORM_PATH)  # [0]
    # if METHOD == "downsample":
    #     new_img = Downsample(
    #         {
    #             "block_size": (1, 2, 2),
    #             "method": "sum",
    #         }
    #     ).run(orig.copy())
    # elif METHOD == "clahe":
    #     new_img = CLAHE(
    #         {
    #             "tileGridSize": (20, 20),
    #             "clipLimit": 3.0,
    #         }
    #     ).run(orig.copy())

    # print(
    #     f"Original: {orig.shape}, Transformed: {transformed.shape}, New: {new_img.shape}"
    # )
    # close = np.allclose(new_img, transformed)
    # print(f"Equivalent? {close}")
    # if not close:
    #     print(new_img.mean(), transformed.mean())
    #     breakpoint()

    # # sk_img = equalize_adapthist(orig, clip_limit=0.5)

    # # Plot all 3 images
    # fig, ax = plt.subplots(1, 3)
    # ax = ax.flatten()
    # ax[0].imshow(orig, cmap="gray")
    # ax[0].set_title("Original")
    # ax[1].imshow(transformed, cmap="gray")
    # ax[1].set_title("ImageJ")
    # ax[2].imshow(new_img, cmap="gray")
    # ax[2].set_title("OpenCV")
    # # ax[3].imshow(sk_img, cmap="gray")
    # # ax[3].set_title("Skimage")
    # fig.tight_layout()
    plt.show()
