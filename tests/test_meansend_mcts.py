import pytest
from chessrl.config.config import MCTSConfig


def test_meansend_alpha_default():
    assert MCTSConfig().meansend_alpha == 0.0


def test_meansend_alpha_settable():
    assert MCTSConfig(meansend_alpha=0.5).meansend_alpha == 0.5


def test_meansend_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        MCTSConfig(meansend_alpha=1.5)
