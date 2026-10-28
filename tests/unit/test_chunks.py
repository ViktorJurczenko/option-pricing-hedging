import pytest

from optionpricing.ml.surrogate.chunks import prepare_chunk_dir


def test_prepare_chunk_dir_accepts_the_same_config_and_rejects_a_different_one(tmp_path):
    chunk_dir = tmp_path / "chunks"
    prepare_chunk_dir(chunk_dir, {"seed": 1, "max_degree": 3})
    prepare_chunk_dir(chunk_dir, {"seed": 1, "max_degree": 3})  # same settings: resumes
    with pytest.raises(ValueError, match="max_degree"):
        prepare_chunk_dir(chunk_dir, {"seed": 1, "max_degree": 4})
