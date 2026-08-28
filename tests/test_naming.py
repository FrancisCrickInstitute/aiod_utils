import pytest

from aiod_utils.io import (
    get_combined_mask_name,
    get_image_id,
    get_mask_name,
    get_mask_prefix,
    get_mask_prefix_from_name,
    is_combined_mask,
    validate_image_ids,
)

# --- get_image_id: dots within the stem, not just the extension ---


def test_get_image_id_ignores_dots_in_stem():
    # Decimal coordinates embedded in the filename look like extra "suffixes"
    # Ensure they are kept and only valid extension identified
    image_id = get_image_id("data_x-2.5_y-1.0_B.ome.tiff")
    assert image_id.stem == "data_x-2.5_y-1.0_B"
    assert image_id.ext == ".ome.tiff"
    assert image_id.value == "data_x-2.5_y-1.0_B_ome_tiff"


def test_get_image_id_dots_in_stem_single_extension():
    assert get_image_id("data_v1.2.3.tif").stem == "data_v1.2.3"
    assert get_image_id("data_v1.2.3.tif").value == "data_v1.2.3_tif"


# --- get_image_id: compound/synthesized extensions still resolve correctly ---


def test_get_image_id_ome_tiff():
    assert get_image_id("cell.ome.tiff").stem == "cell"
    assert get_image_id("cell.ome.tiff").value == "cell_ome_tiff"


def test_get_image_id_ome_zarr_compound():
    # bioio-ome-zarr only registers the bare ".zarr" extension; ".ome.zarr"
    # must still be recognized as the (longer, more specific) compound form.
    assert get_image_id("cell.ome.zarr").ext == ".ome.zarr"
    assert get_image_id("cell.ome.zarr").value == "cell_ome_zarr"


def test_get_image_id_bare_zarr_unaffected():
    assert get_image_id("cell.zarr").ext == ".zarr"
    assert get_image_id("cell.zarr").value == "cell_zarr"


def test_get_image_id_plain_extension():
    assert get_image_id("cell.tif").value == "cell_tif"


def test_get_image_id_ignores_parent_dirs():
    # Identity is derived from the filename only, so the same file reached via
    # different paths (symlink, mount, relative vs absolute) keeps one id
    assert get_image_id("/data/expA/cell.tif") == get_image_id("./cell.tif")


def test_get_image_id_str_is_value():
    # Lets callers interpolate an ImageId straight into a filename
    assert f"{get_image_id('cell.ome.tiff')}" == "cell_ome_tiff"


def test_get_image_id_preserves_extension_case():
    # Distinct files where the filesystem is case-sensitive, so keep them distinct
    assert get_image_id("cell.TIF").value == "cell_TIF"


def test_get_image_id_ome_zarr_needs_no_plugin_registration(monkeypatch):
    # .ome.zarr is added by hand, so it resolves even where no reader plugin
    # registers the bare .zarr it is built from
    import aiod_utils.io as io_mod

    monkeypatch.setattr(io_mod, "get_plugins", lambda use_cache=True: {".tiff": None})
    assert get_image_id("cell.ome.zarr").value == "cell_ome_zarr"


# --- get_image_id: independent of what else is in the run ---


def test_get_image_id_matches_validate_for_colliding_stems():
    # The whole point of folding the extension in unconditionally: a lone
    # get_image_id call cannot disagree with the batch, whatever the batch is
    paths = ["/data/expA/cell.tif", "/data/expA/cell.png"]
    assert validate_image_ids(paths) == [get_image_id(p) for p in paths]


def test_get_image_id_stable_across_batch_composition():
    # Same image, different run composition, same mask filename - otherwise the
    # semi-permanent mask cache would miss on a rerun of a subset
    alone = validate_image_ids(["/data/expA/cell.tif"])[0]
    with_sibling = validate_image_ids(["/data/expA/cell.tif", "/data/expA/cell.png"])[0]
    assert alone == with_sibling


# --- get_image_id: error cases ---


def test_get_image_id_no_extension_raises():
    with pytest.raises(ValueError, match="no extension"):
        get_image_id("cell")


def test_get_image_id_unrecognized_extension_raises():
    with pytest.raises(ValueError, match="unrecognized extension"):
        get_image_id("cell.xyz")


# --- validate_image_ids: distinct names and extensions pass through ---


def test_validate_image_ids_no_conflict():
    paths = ["/data/expA/cell.tif", "/data/expA/nucleus.tif", "/data/expA/mito.png"]
    assert [i.value for i in validate_image_ids(paths)] == [
        "cell_tif",
        "nucleus_tif",
        "mito_png",
    ]


def test_validate_image_ids_single_path():
    assert [i.value for i in validate_image_ids(["/data/expA/cell.tif"])] == [
        "cell_tif"
    ]


def test_validate_image_ids_same_stem_different_extension_passes():
    # No longer a collision to resolve - the extension is always in the id
    paths = ["/data/expA/cell.tiff", "/data/expA/cell.ome.tiff"]
    assert [i.value for i in validate_image_ids(paths)] == [
        "cell_tiff",
        "cell_ome_tiff",
    ]


def test_validate_image_ids_empty():
    assert validate_image_ids([]) == []


# --- validate_image_ids: same name AND extension is unresolvable ---


def test_validate_image_ids_raises_on_same_name_and_extension_different_dirs():
    paths = ["/data/expA/cell.tif", "/data/expB/cell.tif"]
    with pytest.raises(ValueError, match="share both a filename and extension"):
        validate_image_ids(paths)


def test_validate_image_ids_error_names_only_the_colliding_id():
    paths = ["/data/expA/cell.tif", "/data/expB/cell.tif", "/data/expC/cell.png"]
    with pytest.raises(ValueError, match="cell_tif") as excinfo:
        validate_image_ids(paths)
    assert "cell_png" not in str(excinfo.value)


def test_validate_image_ids_identical_path_listed_twice_raises():
    paths = ["/data/expA/cell.tif", "/data/expA/cell.tif"]
    with pytest.raises(ValueError, match="share both a filename and extension"):
        validate_image_ids(paths)


# --- get_mask_name: consumes the ImageId ---


def test_get_mask_name_from_image_id():
    image_id = get_image_id("cell.ome.tiff")
    assert get_mask_name(run_hash="1a2b3c4d", image_id=image_id) == (
        "cell_ome_tiff_masks_1a2b3c4d"
    )


def test_get_mask_name_from_image_path_matches_image_id():
    # The two entry points must not drift - napari uses one, callers with only
    # a path use the other
    path = "/data/expA/cell.ome.tiff"
    assert get_mask_name(run_hash="1a2b3c4d", image_path=path) == get_mask_name(
        run_hash="1a2b3c4d", image_id=get_image_id(path)
    )


def test_get_mask_name_with_prep_hash():
    assert (
        get_mask_name(
            run_hash="1a2b3c4d", image_id=get_image_id("cell.tif"), prep_hash="deadbeef"
        )
        == "cell_tif_deadbeef_masks_1a2b3c4d"
    )


def test_get_mask_name_distinguishes_same_stem_different_extension():
    # The failure this whole scheme exists to prevent: two source images
    # writing masks to the same filename
    assert get_mask_name(
        run_hash="1a2b3c4d", image_path="/data/cell.tif"
    ) != get_mask_name(run_hash="1a2b3c4d", image_path="/data/cell.png")


def test_get_mask_name_requires_an_image():
    with pytest.raises(ValueError, match="must be provided"):
        get_mask_name(run_hash="1a2b3c4d")


# --- get_mask_prefix: the (image, preprocessing) identity ---


def test_get_mask_prefix_is_the_mask_name_without_the_run_hash():
    # The two must not drift: consumers match masks on disk by this prefix
    image_id = get_image_id("cell.ome.tiff")
    for prep_hash in (None, "deadbeef"):
        prefix = get_mask_prefix(image_id, prep_hash=prep_hash)
        name = get_mask_name(
            run_hash="1a2b3c4d", image_id=image_id, prep_hash=prep_hash
        )
        assert name == f"{prefix}_masks_1a2b3c4d"


def test_get_mask_prefix_carries_the_full_image_id():
    assert get_mask_prefix(get_image_id("cell.ome.tiff")) == "cell_ome_tiff"


def test_get_mask_prefix_distinguishes_preprocessing_sets():
    image_id = get_image_id("cell.tif")
    assert get_mask_prefix(image_id) != get_mask_prefix(image_id, prep_hash="deadbeef")


def test_get_mask_prefix_accepts_a_plain_string_id():
    assert get_mask_prefix("cell_tif", prep_hash="deadbeef") == "cell_tif_deadbeef"


# --- mask name construction and parsing round-trip ---


def test_mask_prefix_recovered_from_a_substack_filename():
    # What a directory watcher has to do: it only sees filenames
    name = get_mask_name(run_hash="1a2b3c4d", image_path="/data/cell.ome.tiff")
    written = f"{name}_x0-64_y0-64_z0-1.rle"
    assert get_mask_prefix_from_name(written) == get_mask_prefix(
        get_image_id("cell.ome.tiff")
    )


def test_mask_prefix_recovered_from_a_combined_filename():
    name = get_mask_name(run_hash="1a2b3c4d", image_path="/data/cell.tif")
    assert get_mask_prefix_from_name(get_combined_mask_name(name, "rle")) == "cell_tif"


def test_mask_prefix_survives_a_separator_in_the_image_name():
    # rsplit, not split: this image's own name contains the separator
    name = get_mask_name(run_hash="1a2b3c4d", image_path="/data/foo_masks_bar.tif")
    assert get_mask_prefix_from_name(f"{name}_x0-64_y0-64_z0-1.rle") == (
        "foo_masks_bar_tif"
    )


def test_combined_mask_name_with_and_without_extension():
    assert get_combined_mask_name("cell_tif_masks_1a2b") == "cell_tif_masks_1a2b_all"
    assert get_combined_mask_name("cell_tif_masks_1a2b", "rle") == (
        "cell_tif_masks_1a2b_all.rle"
    )


def test_is_combined_mask_distinguishes_substacks():
    name = get_mask_name(run_hash="1a2b3c4d", image_path="/data/cell.tif")
    assert is_combined_mask(get_combined_mask_name(name, "rle"))
    assert not is_combined_mask(f"{name}_x0-64_y0-64_z0-1.rle")
