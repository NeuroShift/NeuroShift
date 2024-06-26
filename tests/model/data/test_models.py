from neuroshift.model.data.models import Models
from neuroshift.model.data.model import Model


def test_models_get_instance() -> None:
    models = Models.get_instance()

    assert models is not None
    assert isinstance(models, Models)
    assert models == Models.get_instance()


def test_models_select(mnist_models: Models, mnist_model: Model) -> None:
    first_model = mnist_models.get_models()[0]

    assert mnist_models.get_selected() == first_model

    mnist_models.select(mnist_model)

    assert mnist_models.get_selected() == mnist_model
    assert mnist_model.is_selected()

    mnist_models.select(None)
    assert mnist_models.get_selected() == first_model


def test_models_get_models(mnist_model: Model) -> None:
    models = Models.get_instance()

    model_list = models.get_models()
    assert len(model_list) != 0

    models.add(mnist_model)
    assert len(model_list) + 1 == len(models.get_models())
    assert model_list + [mnist_model] == models.get_models()


def test_models_get_model_names(mnist_model: Model) -> None:
    models = Models.get_instance()

    model_names = models.get_model_names()
    assert len(model_names) != 0

    models.add(mnist_model)
    assert len(model_names) + 1 == len(models.get_model_names())
    assert model_names + [mnist_model.get_name()] == models.get_model_names()


def test_models_delete(mnist_model: Model) -> None:
    models = Models.get_instance()

    model_list = models.get_models()
    models.add(mnist_model)

    for model in model_list:
        if model == mnist_model:
            continue

        models.delete(model)

    assert len(models.get_models()) == 1
    assert models.get_models() == [mnist_model]
