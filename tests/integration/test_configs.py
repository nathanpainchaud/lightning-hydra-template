import hydra
import pytest
from _pytest.fixtures import FixtureRequest
from hydra.core.hydra_config import HydraConfig


@pytest.mark.parametrize("cfg_fixture_name", ["cfg_train", "cfg_eval"])
def test_config(request: FixtureRequest, cfg_fixture_name: str) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    Args:
        request: The pytest request special fixture.
        cfg_fixture_name: The name of the configuration fixture to be tested.
    """
    assert cfg_fixture_name

    cfg = request.getfixturevalue(cfg_fixture_name)

    assert cfg.data
    assert cfg.model
    assert cfg.trainer

    HydraConfig().set_config(cfg)

    hydra.utils.instantiate(cfg.data)
    hydra.utils.instantiate(cfg.model)
    hydra.utils.instantiate(cfg.trainer)
