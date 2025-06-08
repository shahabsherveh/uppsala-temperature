from shastat.model import StructuralTimeSeriesBuilder
import pytest


def test_model_initialization():
    """
    Test the initialization of the StructuralTimeSeriesBuilder model.
    """
    model = StructuralTimeSeriesBuilder()
    assert model is not None, "Model should be initialized successfully."
    assert hasattr(model, "_model_skeleton"), "Model skeleton should be built."
    assert model._model_type == "StructuralTimeSeries", "Model type should match."
    # Check if the model skeleton is a valid object
    assert model._model_skeleton is not None, "Model skeleton should not be None."
    assert hasattr(model, "_ar"), "Model should have an autoregressive component."


def test_model_fit():
    """
    Test the fit method of the StructuralTimeSeriesBuilder model.
    """
    model = StructuralTimeSeriesBuilder()
    # Create a dummy time series data
    import pandas as pd
    import numpy as np

    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    y = pd.Series(np.random.randn(10), index=dates)

    # Fit the model with the dummy data
    model.fit(y=y)

    assert hasattr(model, "is_fitted_"), "Model should have a fitted attribute."
    assert model.is_fitted_ is True, "Model should be marked as fitted."
