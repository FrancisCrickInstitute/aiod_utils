import warnings
from collections import namedtuple
from math import floor
from typing import Union

from numpy import dtype as np_dtype

# TODO: Handle the case where the image is 2D (slice) or 3D (slice with channels)
# TODO: Handle the case where the image is 4D (volume + channels)
# TODO: Handle the case where the image is 5D (volume + time + channels)
# Default channels to None, as not always needed
Stack = namedtuple("Stack", ["height", "width", "depth", "channels"], defaults=[None])
# TODO: These constants vary by GPU/available memory, and also scale by number of channels which is not considered here
# Named substack to avoid patch/tile confusion, and to avoid chunk confusion
MAX_SUBSTACK_SIZE = Stack(
    height=5000,
    width=5000,
    depth=50,
    channels=1_000_000,  # effectively unconstrained;
)


HEADROOM_FACTOR = 0.75


def compute_max_substack_size(
    memory_bytes: int, dtype: str, image_shape: Stack
) -> Stack:
    """
    Compute the maximum substack size (in pixels) given available memory,
    preserving the aspect ratio of the image.

    Scaling dims proportionally ensures the memory budget is fully used rather
    than being wasted when one dimension (e.g. depth) is much smaller than the
    cubic root of the voxel budget.

    Returns MAX_SUBSTACK_SIZE when image_shape has a non-positive total voxel
    count. Raises ValueError if the memory budget is too small to allocate even
    a single voxel after applying HEADROOM_FACTOR.
    """

    usable = memory_bytes * HEADROOM_FACTOR
    # TODO worry about bitdepth? RGB channels maybe a special case, check consistency with the whole "S is C" business ... 0.0
    bytes_per_voxel = np_dtype(dtype).itemsize * (image_shape.channels or 1)
    max_voxels = usable / bytes_per_voxel
    total_voxels = image_shape.height * image_shape.width * image_shape.depth
    if total_voxels <= 0:
        return MAX_SUBSTACK_SIZE
    if max_voxels < 1:
        raise ValueError(
            f"Memory budget too small: {memory_bytes} bytes with dtype={dtype} and "
            f"n_channels={image_shape.channels} gives {max_voxels:.2f} usable voxels after "
            f"applying HEADROOM_FACTOR={HEADROOM_FACTOR}. Cannot allocate even a "
            f"single voxel. Check params.memory_per_job in your Nextflow profile."
        )
    scale = (max_voxels / total_voxels) ** (1 / 3)
    if scale >= 1.0:
        # Whole image fits in one substack
        return image_shape
    h = max(1, floor(image_shape.height * scale))
    w = max(1, floor(image_shape.width * scale))
    d = max(1, floor(image_shape.depth * scale))
    return Stack(height=h, width=w, depth=d, channels=image_shape.channels)


def auto_size(size: int, max_size: Union[int, float]) -> int:
    """
    Calculate the number of stacks to use for a given size, based on a maximum size.
    """
    num_stacks = size // max_size
    # If the size is less than the max size, we still want to use 1 stack
    if num_stacks == 0:
        return 1
    # We want to ensure that max_size is not exceeded
    if (size / num_stacks) > max_size:
        num_stacks += 1
    return int(num_stacks)


def calc_num_stacks_dim(
    dim_size: int,
    req_stacks_dim: int | str,
    overlap: float,
    dim: str,
    max_substack_size: Stack = MAX_SUBSTACK_SIZE,
) -> tuple[int, int]:
    """
    Determine the number of stacks to use for a given dimension, based on the size of the dimension, the number of stacks requested, the overlap fraction requested, and our maximum size constraints set at the top.

    If max_substack_size is provided (e.g. computed from available memory), it is used
    instead of the global MAX_SUBSTACK_SIZE constant.
    """
    cap = getattr(max_substack_size, dim)
    # Calculate the effective size of the image after multiplying by the added overlap
    # This is ignored/irrelevant if only 1 stack is requested or if the overlap is 0 respectively
    eff_size = round(dim_size * (1 + overlap))

    if req_stacks_dim == "auto":
        num_stacks_dim = auto_size(eff_size, cap)
    else:
        num_stacks_dim = max(1, int(req_stacks_dim))
        # Check that whatever num_stacks_dim set is sensible
        # More stacks than pixels? Use auto method
        if num_stacks_dim > eff_size:
            num_stacks_dim = auto_size(eff_size, cap)
        # Make sure it's not too small, defined as 5% of the max stacks size
        elif (eff_size // num_stacks_dim) < (cap / 20):
            # NOTE: We are just ignoring the user here
            # We could default to cap / 20 as a lower bound
            # But I think I'd rather just set our default stacks size and ignore it!
            num_stacks_dim = auto_size(eff_size, cap)
        # Make sure it's not too big, defined as 3x the max stacks size
        elif (eff_size // num_stacks_dim) > (cap * 3):
            # NOTE: Again we are just ignoring the user here
            # We could default to cap * 3 as an upper bound...
            num_stacks_dim = auto_size(eff_size, cap)

    # If num_stacks_dim is 1, and overlap is >0, warn the user
    if num_stacks_dim == 1 and overlap > 0:
        warnings.warn(
            f"Ignoring overlap setting {overlap} for dimension {dim} with only 1 stack"
            + (" (calculated automatically)" if req_stacks_dim == "auto" else "")
        )
    return num_stacks_dim, eff_size


def calc_num_stacks(
    image_shape: Stack,
    req_stacks: Stack,
    overlap_fraction: Stack,
    max_substack_size: Stack = MAX_SUBSTACK_SIZE,
) -> tuple[Stack, Stack]:
    num_stacks_height, eff_height = calc_num_stacks_dim(
        image_shape.height,
        req_stacks.height,
        overlap_fraction.height,
        "height",
        max_substack_size,
    )
    num_stacks_width, eff_width = calc_num_stacks_dim(
        image_shape.width,
        req_stacks.width,
        overlap_fraction.width,
        "width",
        max_substack_size,
    )
    num_stacks_depth, eff_depth = calc_num_stacks_dim(
        image_shape.depth,
        req_stacks.depth,
        overlap_fraction.depth,
        "depth",
        max_substack_size,
    )
    num_stacks = Stack(
        height=num_stacks_height,
        width=num_stacks_width,
        depth=num_stacks_depth,
        channels=image_shape.channels,
    )
    eff_shape = Stack(
        height=eff_height,
        width=eff_width,
        depth=eff_depth,
        channels=image_shape.channels,
    )
    return num_stacks, eff_shape


def generate_stack_indices(
    image_shape: Stack,
    num_substacks: Stack,
    overlap_fraction: Stack,
    eff_shape: Stack,
) -> tuple[list[tuple[tuple[int, int], ...]], int, Stack]:
    """
    Generate the indices for every stack for a given image size, desired number of substacks, and overlap fraction.

    Note that the overlap fraction is a float between 0 and 1, and the number of substacks is a tuple of integers, both of which should represent the same number of dimensions and meaning of the image_shape, which is expected to be a tuple of integers representing the dimensions of the image (D, H, W).

    Also note that the output is not guaranteed to completely satisfy the given arguments, as it may not be satisfiable. In this case, the overlap created will be different, but the number of substacks is guaranteed.
    """
    # Calculate the stack size based on the number of substacks
    stack_height = eff_shape.height // num_substacks.height
    stack_width = eff_shape.width // num_substacks.width
    stack_depth = eff_shape.depth // num_substacks.depth

    # Calculate overlap size based on fraction
    overlap_height = (
        0
        if overlap_fraction.height == 0
        else max(int(stack_height * overlap_fraction.height), 1)
    )
    overlap_width = (
        0
        if overlap_fraction.width == 0
        else max(int(stack_width * overlap_fraction.width), 1)
    )
    overlap_depth = (
        0
        if overlap_fraction.depth == 0
        else max(int(stack_depth * overlap_fraction.depth), 1)
    )

    # Generate indices for the substacks
    stack_indices = []

    for i in range(num_substacks.height):
        for j in range(num_substacks.width):
            for k in range(num_substacks.depth):
                start_h = i * (stack_height - overlap_height)
                end_h = (
                    min(start_h + stack_height, image_shape.height)
                    if i < max(num_substacks.height, 1) - 1
                    else image_shape.height
                )
                start_w = j * (stack_width - overlap_width)
                end_w = (
                    min(start_w + stack_width, image_shape.width)
                    if j < max(num_substacks.width, 1) - 1
                    else image_shape.width
                )
                start_d = k * (stack_depth - overlap_depth)
                end_d = (
                    min(start_d + stack_depth, image_shape.depth)
                    if k < max(num_substacks.depth, 1) - 1
                    else image_shape.depth
                )

                stack_indices.append(
                    ((start_h, end_h), (start_w, end_w), (start_d, end_d))
                )
    return (
        stack_indices,
        len(stack_indices),
        Stack(stack_height, stack_width, stack_depth, image_shape.channels),
    )


def check_sizes(stack_indices):
    "Quick check to see if the stack sizes are all the same."
    sizes = []
    for stack in stack_indices:
        stack_size = 0
        for stack_dim in stack:
            stack_size += stack_dim[1] - stack_dim[0]
        sizes.append(stack_size)
    if len(set(sizes)) == 1:
        return True
    else:
        return False
