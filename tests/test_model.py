from shastat.model import StructuralTimeSeriesBuilder
import pandas as pd
import numpy as np
import pytest


class TestDefaultModel:
    model = StructuralTimeSeriesBuilder()

    def test_model_initialization(self):
        """
        Test the initialization of the StructuralTimeSeriesBuilder model.
        """
        model = self.model
        assert model is not None, "Model should be initialized successfully."
        assert hasattr(model, "_model_skeleton"), "Model skeleton should be built."
        assert model._model_type == "StructuralTimeSeries", "Model type should match."
        # Check if the model skeleton is a valid object
        assert model._model_skeleton is not None, "Model skeleton should not be None."
        assert hasattr(model, "_ar"), "Model should have an autoregressive component."

    def test_model_fit(self):
        """
        Test the fit method of the StructuralTimeSeriesBuilder model.
        """
        model = self.model
        # Create a dummy time series data
        import pandas as pd
        import numpy as np

        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        y = pd.Series(np.random.randn(10), index=dates)

        # Fit the model with the dummy data
        model.fit(y=y)

        assert hasattr(model, "is_fitted_"), "Model should have a fitted attribute."
        assert model.is_fitted_ is True, "Model should be marked as fitted."

    def test_model_save(self):
        """
        Test the save method of the StructuralTimeSeriesBuilder model.
        """
        model = self.model
        # Save the model to a file
        # Check if the file exists
        import os

        dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
        y = pd.Series(np.random.randn(10), index=dates)

        # Fit the model with the dummy data
        model.fit(y=y)

        model.save("tests/test_data/model.nc")
        assert os.path.exists("tests/test_data/model.nc"), "Model file should be saved."

    def test_model_load(self):
        """
        Test the load method of the StructuralTimeSeriesBuilder model.

        """
        model = StructuralTimeSeriesBuilder.load("tests/test_data/model.nc")
        model.forecast(steps=5)
