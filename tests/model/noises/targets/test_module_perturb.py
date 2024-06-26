import pytest

from neuroshift.model.noises.perturbation import Perturbation
from neuroshift.model.data.model import Model
from neuroshift.model.noises.targets.module_perturb import ModulePerturb
from tests.model.noises.test_perturbation import perturbation  # noqa


@pytest.fixture
def module_perturb(
    mnist_model: Model, perturbation: Perturbation
) -> ModulePerturb:
    return ModulePerturb(
        module=mnist_model.get_model(), perturbation=perturbation
    )


def test_constructor(module_perturb: ModulePerturb) -> None:
    assert isinstance(module_perturb, ModulePerturb)


def test_module_perturb_run(
    module_perturb: ModulePerturb, mnist_model: Model
) -> None:
    perturbed_module = module_perturb.run()

    assert perturbed_module is not mnist_model.get_model()
