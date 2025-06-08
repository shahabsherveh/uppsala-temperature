import warnings
from typing import Any
import pymc as pm
from pymc_extras.statespace import structural as st
from pymc_extras.model_builder import ModelBuilder
from pymc.util import RandomState
import arviz as az
from pytensor import tensor as pt
import pandas as pd
import numpy as np


class BaseConfig:
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class SamplerConfig(BaseConfig):
    def __init__(self, n_chains=4, n_samples=1000, n_burn=500):
        self.n_chains = n_chains
        self.n_samples = n_samples
        self.n_burn = n_burn


class TrendConfig(BaseConfig):
    def __init__(self, order, innovations_order):
        self.order = order
        self.innovations_order = innovations_order


class ARConfig(BaseConfig):
    def __init__(self, order):
        self.order = order


class TimeSeasonalConfig(BaseConfig):
    def __init__(self, season_length, inovation, name):
        self.season_length = season_length
        self.innovation = inovation
        self.name = name


class FrequencySeasonalityConfig(BaseConfig):
    def __init__(self, season_length, n, innovation, name):
        self.season_length = season_length
        self.n = n
        self.name = name
        self.innovation = innovation


class StructuralTimeSeriesConfig:
    def __init__(
        self,
        trend_order: int | None = None,
        trend_innovations_order: int = 0,
        ar_order: int | None = 1,
        season_length: int | None = None,
        season_innovation: bool = True,
        seasonal_name: str = "annual",
        cycle_length: int | list[int] | None = None,
        cycle_n: int | list[int] | None = None,
        cycle_innovation: bool = True,
    ):
        if trend_order is not None:
            self.trend = TrendConfig(trend_order, trend_innovations_order)
        if ar_order is not None:
            self.ar = ARConfig(ar_order)
        if season_length is not None:
            self.seasonal = TimeSeasonalConfig(
                season_length, season_innovation, seasonal_name
            )
        if cycle_length is not None:
            cycle_length = (
                cycle_length if isinstance(cycle_length, list) else [cycle_length]
            )
            self.cycles = (
                FrequencySeasonalityConfig(
                    season_length, cycle_n, cycle_innovation, "cycle"
                )
                if cycle_length
                else None
            )

    def to_dict(self):
        components = [k for k in self.__dict__ if not k.startswith("_")]
        config = {}
        for k in components:
            v = getattr(self, k)
            if isinstance(v, list):
                config[k] = [item.to_dict() for item in v]
            elif v:
                config[k] = v.to_dict()
            else:
                config[k] = None
        return config


class StructuralTimeSeriesBuilder(ModelBuilder):
    _model_type = "StructuralTimeSeries"

    def __init__(
        self,
        model_config: StructuralTimeSeriesConfig | None = None,
        sampler_config: SamplerConfig | None = None,
    ):
        if model_config is None:
            model_config = StructuralTimeSeriesConfig()
        if sampler_config is None:
            sampler_config = SamplerConfig()
        super().__init__(model_config.to_dict(), sampler_config.to_dict())
        self._model_skeleton = self.build_skeleton()

    def build_skeleton(self):
        ar_config = self.model_config.get("ar")
        trend_config = self.model_config.get("trend")
        seasonal_config = self.model_config.get("seasonal")
        cycle_config = self.model_config.get("cycles")

        if ar_config:
            self._ar = st.AutoregressiveComponent(**ar_config)
            self._hidden_state = self._ar
        if trend_config:
            self._trend = (
                st.LevelTrendComponent(**trend_config) if trend_config else None
            )
            self._hidden_state += self._trend
        if seasonal_config:
            self._seasonals = (
                st.TimeSeasonality(**seasonal_config) if seasonal_config else None
            )
            self._hidden_state += self._seasonals

        if cycle_config:
            self._cycles = [st.FrequencySeasonality(**c) for c in cycle_config]
            for cycle in self._cycles:
                self._hidden_state += cycle

        return self._hidden_state.build()

    def build_priors(self, param_names, param_dims, k_states) -> None:
        for name in param_names:
            if name == "P0":
                P0 = pm.Deterministic(
                    "P0",
                    pt.eye(k_states),
                    dims=param_dims["P0"],
                )
            elif name.startswith("sigma_"):
                locals()[name] = pm.Exponential(name, dims=param_dims.get(name))
            else:
                locals()[name] = pm.Normal(name, dims=param_dims[name])

    def build_model(
        self, X: pd.DataFrame | pd.Series | None, y: pd.Series | pd.DataFrame, **kwargs
    ) -> None:
        self._generate_and_preprocess_model_data(X, y)
        param_names = self._model_skeleton.param_names
        param_dims = self._model_skeleton.param_dims
        k_states = self._model_skeleton.k_states
        print(f"Building model with parameters: {param_names}")
        print(f"Parameter dimensions: {param_dims}")

        with pm.Model(coords=self._model_skeleton.coords) as self.model:
            self.build_priors(param_names, param_dims, k_states)
            self._model_skeleton.build_statespace_graph(data=y.to_frame())

    def _generate_and_preprocess_model_data(
        self,
        X: pd.DataFrame | pd.Series | None = None,
        y: pd.DataFrame | pd.Series = pd.Series(),
    ) -> None:
        self._X = X
        self._y = y

    def _data_setter(self, X=None, y=None):
        if isinstance(X, pd.DataFrame):
            x_values = X["input"].values
        else:
            # Assuming "input" is the first column
            x_values = X[:, 0]

        with self.model:
            pm.set_data({"x_data": x_values})
            if y is not None:
                pm.set_data({"y_data": y})

    @property
    def output_var(self):
        return "y"

    def fit(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series = pd.Series(),
        progressbar: bool = True,
        predictor_names: list[str] | None = None,
        random_seed: RandomState = None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """
        Fit a model using the data passed as a parameter.
        Sets attrs to inference data of the model.


        Parameters
        ----------
        X : array-like if sklearn is available, otherwise array, shape (n_obs, n_features)
            The training input samples.
        y : array-like if sklearn is available, otherwise array, shape (n_obs,)
            The target values (real numbers).
        progressbar : bool
            Specifies whether the fit progressbar should be displayed
        predictor_names: List[str] = None,
            Allows for custom naming of predictors given in a form of 2dArray
            allows for naming of predictors when given in a form of np.ndarray, if not provided the predictors will be named like predictor1, predictor2...
        random_seed : RandomState
            Provides sampler with initial random seed for obtaining reproducible samples
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            returns inference data of the fitted model.

        Examples
        --------
        >>> model = MyModel()
        >>> idata = model.fit(data)
        Auto-assigning NUTS sampler...
        Initializing NUTS using jitter+adapt_diag...
        """
        self._generate_and_preprocess_model_data(X, y)
        self.build_model(self._X, self._y)

        sampler_config = self.sampler_config.copy()
        sampler_config["progressbar"] = progressbar
        sampler_config["random_seed"] = random_seed
        sampler_config.update(**kwargs)
        self.idata = self.sample_model(**sampler_config)

        if isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=X.columns)
            combined_data = pd.concat([X_df, y], axis=1)
            assert all(combined_data.columns), "All columns must have non-empty names"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The group fit_data is not defined in the InferenceData scheme",
                )
                self.idata.add_groups(fit_data=combined_data.to_xarray())  # type: ignore

        self.is_fitted_ = True
        return self.idata  # type: ignore

    def forecast(
        self,
        steps: int,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Forecast future values based on the fitted model.
        Parameters
        ----------
        steps : int
            Number of steps to forecast.
        X : pd.DataFrame, optional
            Additional input data for forecasting.
        y : pd.Series or pd.DataFrame, optional
            Target values for forecasting.
        Returns
        -------
        pd.DataFrame
            Forecasted values.
        """
        assert self.is_fitted_, "Model must be fitted before forecasting."
        return self._model_skeleton.forecast(idata=self.idata, periods=steps, **kwargs)
