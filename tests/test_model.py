from src.models.model import MyAwesomeModel
import torch
import pytest

class TestClass:
    model = MyAwesomeModel([64, 32, 16, 8], 128)

    def test_model(self):
        ts = torch.rand((1, 1, 28, 28))
        output = self.model(ts)

        assert output.shape == (1, 10), "Output shape should be (1, 10)"

    def test_error_on_wrong_shape(self):
        with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
            self.model(torch.randn(1, 2, 3))
        with pytest.raises(ValueError, match='Expected each sample to have shape 1, 28, 28'):
            self.model(torch.rand((1, 2, 28, 28)))
