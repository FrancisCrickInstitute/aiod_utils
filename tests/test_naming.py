import pytest

from aiod_utils.io import get_image_id, resolve_image_ids

# --- get_image_id: dots within the stem, not just the extension ---


def test_get_image_id_ignores_dots_in_stem():
    # Decimal coordinates embedded in the filename look like extra "suffixes"
    # Ensure they are kept and only valid extension identified
    assert get_image_id("data_x-2.5_y-1.0_B.ome.tiff") == "data_x-2.5_y-1.0_B"


def test_get_image_id_dots_in_stem_single_extension():
    assert get_image_id("data_v1.2.3.tif") == "data_v1.2.3"


# --- get_image_id: compound/synthesized extensions still resolve correctly ---


def test_get_image_id_ome_tiff():
    assert get_image_id("cell.ome.tiff") == "cell"


def test_get_image_id_ome_zarr_synthesized():
    # bioio-ome-zarr only registers the bare ".zarr" extension; ".ome.zarr"
    # must still be recognized as the (longer, more specific) compound form.
    assert get_image_id("cell.ome.zarr") == "cell"


def test_get_image_id_bare_zarr_unaffected():
    assert get_image_id("cell.zarr") == "cell"


def test_get_image_id_plain_extension():
    assert get_image_id("cell.tif") == "cell"


# --- get_image_id: error cases ---


def test_get_image_id_no_extension_raises():
    with pytest.raises(ValueError, match="no extension"):
        get_image_id("cell")


def test_get_image_id_unrecognized_extension_raises():
    with pytest.raises(ValueError, match="unrecognized extension"):
        get_image_id("cell.xyz")


# --- No conflicts: ids stay bare stems ---


def test_resolve_image_ids_no_conflict():
    paths = ["/data/expA/cell.tif", "/data/expA/nucleus.tif", "/data/expA/mito.png"]
    assert resolve_image_ids(paths) == ["cell", "nucleus", "mito"]


def test_resolve_image_ids_single_path():
    assert resolve_image_ids(["/data/expA/cell.tif"]) == ["cell"]


# --- Same stem, different extension: disambiguated by extension ---


def test_resolve_image_ids_disambiguates_by_extension():
    paths = ["/data/expA/cell.tif", "/data/expA/cell.png"]
    assert resolve_image_ids(paths) == ["cell_tif", "cell_png"]


def test_resolve_image_ids_disambiguates_multidot_extension():
    paths = ["/data/expA/cell.ome.tiff", "/data/expA/cell.tif"]
    assert resolve_image_ids(paths) == ["cell_ome_tiff", "cell_tif"]


# --- Same stem AND extension from different directories: unresolvable ---


def test_resolve_image_ids_raises_on_same_name_and_extension_different_dirs():
    paths = ["/data/expA/cell.tif", "/data/expB/cell.tif"]
    with pytest.raises(ValueError, match="share both a filename and extension"):
        resolve_image_ids(paths)


def test_resolve_image_ids_mixed_group_only_fails_the_genuine_collision():
    # cell.tif clashes across dirs even after extension disambiguation;
    # cell.png is a distinct extension and resolves cleanly alongside them.
    paths = ["/data/expA/cell.tif", "/data/expB/cell.tif", "/data/expC/cell.png"]
    with pytest.raises(ValueError, match="cell_tif"):
        resolve_image_ids(paths)


def test_resolve_image_ids_identical_path_listed_twice_raises():
    paths = ["/data/expA/cell.tif", "/data/expA/cell.tif"]
    with pytest.raises(ValueError, match="share both a filename and extension"):
        resolve_image_ids(paths)
