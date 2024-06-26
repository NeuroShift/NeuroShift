import torch
import pytest
from neuroshift.model.noises.model_distribution_shift.stuck_at_fault import (
    StuckAtFault,
)


@pytest.fixture
def stuck_at_fault() -> None:
    return StuckAtFault.get_instance()


def test_constructor(stuck_at_fault: StuckAtFault) -> None:
    assert isinstance(stuck_at_fault, StuckAtFault)


def test_get_instance(stuck_at_fault: StuckAtFault) -> None:
    another_instance = stuck_at_fault.get_instance()

    assert another_instance is stuck_at_fault


def test_apply_to_tensor(stuck_at_fault: StuckAtFault) -> None:
    stuck_at_fault.get_parameters()[0].set_value(1.0)
    tensor = torch.rand(256, 256)

    perturbed_tensor = stuck_at_fault.apply_to_tensor(tensor)

    assert tensor.shape == perturbed_tensor.shape
    assert not torch.all(torch.eq(tensor, perturbed_tensor))


def test_apply_to_tensor_with_no_noise(stuck_at_fault: StuckAtFault) -> None:
    stuck_at_fault.get_parameters()[0].set_value(0)
    tensor = torch.rand(256, 256)

    perturbed_tensor = stuck_at_fault.apply_to_tensor(tensor)

    assert tensor.shape == perturbed_tensor.shape
    assert torch.all(torch.eq(tensor, perturbed_tensor))
