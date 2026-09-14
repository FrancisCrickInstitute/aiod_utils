import aiod_utils.preprocess

# ── get_downsample_factor ─────────────────────────────────────────────────────


def test_get_downsample_factor_from_methods():
    methods = [
        {"name": "Downsample", "params": {"block_size": [10, 10, 10], "method": "mean"}}
    ]
    assert aiod_utils.preprocess.get_downsample_factor(methods=methods) == (10, 10, 10)


def test_get_downsample_factor_from_methods_no_downsample():
    methods = [{"name": "GaussianBlur", "params": {"sigma": 1.0}}]
    assert aiod_utils.preprocess.get_downsample_factor(methods=methods) is None


def test_get_downsample_factor_from_empty_methods():
    assert aiod_utils.preprocess.get_downsample_factor(methods=[]) is None


# --- get_prep_hash: one definition of the prep_set -> hash rule ---


def test_get_prep_hash_matches_the_two_step_it_replaces():
    methods = [{"name": "CLAHE", "params": {"clipLimit": 3.0, "tileGridSize": [8, 8]}}]
    params_str = aiod_utils.preprocess.get_params_str(methods, to_save=True)
    assert aiod_utils.preprocess.get_prep_hash(
        methods
    ) == aiod_utils.preprocess.hash_params_str(params_str)


def test_get_prep_hash_is_none_for_a_no_op_set():
    # Every caller used to guard this by hand
    assert aiod_utils.preprocess.get_prep_hash([]) is None
    assert aiod_utils.preprocess.get_prep_hash(None) is None


def test_get_prep_hash_distinguishes_different_params():
    a = [{"name": "CLAHE", "params": {"clipLimit": 3.0, "tileGridSize": [8, 8]}}]
    b = [{"name": "CLAHE", "params": {"clipLimit": 5.0, "tileGridSize": [8, 8]}}]
    assert aiod_utils.preprocess.get_prep_hash(
        a
    ) != aiod_utils.preprocess.get_prep_hash(b)
