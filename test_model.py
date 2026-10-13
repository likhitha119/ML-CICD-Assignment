
from model import train_model
from sklearn.datasets import load_iris

def test_model_training():
    model, accuracy = train_model()
    assert model is not None
    assert 0 <= accuracy <= 1

def test_model_predictions():
    model, accuracy = train_model()
    iris = load_iris()
    predictions = model.predict(iris.data[:5])

    assert len(predictions) == 5
    assert all(p in [0, 1, 2] for p in predictions)
