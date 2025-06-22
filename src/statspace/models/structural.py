import numpy as np
from .utils import BaseParams


class FrequencySeasonalityParams(BaseParams):
    def __init__(self, period, harmonics, stochastic):
        self.period = period
        self.harmonics = harmonics
        self.stochastic = stochastic

    @classmethod
    def from_string(cls, freq_seasonal_str: str):
        """
        Create a FrequencySeasonalityParams instance from a string.
        The string should be in the format 'period=<period>,harmonics=<harmonics>,stochastic=<stochastic>'.
        """
        parts = freq_seasonal_str.split(",")
        params = {}
        for part in parts:
            key, value = part.split("=")
            params[key.strip()] = value.strip()
        return cls(
            period=int(params["period"]),
            harmonics=int(params["harmonics"]),
            stochastic=params["stochastic"].lower() == "true",
        )


class FrequencySeasonalityParamsList(BaseParams):
    def __init__(self, frequency_seasonalities: list[FrequencySeasonalityParams] = []):
        self.frequency_seasonalities = frequency_seasonalities

    def parse(self):
        d_output = dict(freq_seasonal=[], stochastic_freq_seasonal=[])
        for c in self.frequency_seasonalities:
            d = c.to_dict()
            d_output["stochastic_freq_seasonal"].append(d.pop("stochastic"))
            d_output["freq_seasonal"].append(d)
        return d_output

    @classmethod
    def from_strings(cls, freq_seasonal_strs: list[str]):
        """
        Create a FrequencySeasonalityParamsList instance from a list of strings.
        Each string should be in the format 'period=<period>,harmonics=<harmonics>,stochastic=<stochastic>'.
        """
        frequency_seasonalities = [
            FrequencySeasonalityParams.from_string(fs) for fs in freq_seasonal_strs
        ]
        return cls(frequency_seasonalities=frequency_seasonalities)


class CyclePeriodBoundsParams(BaseParams):
    def __init__(self, lower=0.5, upper=np.inf):
        self.lower = lower
        self.upper = upper

    def parse(self):
        return (self.lower, self.upper)

    @classmethod
    def from_string(cls, bounds_str: str):
        """
        Create a CyclePeriodBoundsParams instance from a string.
        The string should be in the format 'lower=<lower>,upper=<upper>'.
        """
        parts = bounds_str.split(",")
        params = {}
        for part in parts:
            key, value = part.split("=")
            params[key.strip()] = (
                float(value.strip()) if value.strip().lower() != "infinity" else np.inf
            )
        return cls(lower=params["lower"], upper=params["upper"])


class StructuralTimeSeriesParams(BaseParams):
    def __init__(
        self,
        autoregressive: int | None = None,
        trend: bool | None = None,
        level: bool | None = None,
        seasonal: int | None = None,
        freq_seasonal: FrequencySeasonalityParamsList = FrequencySeasonalityParamsList(),
        cycle: bool | None = None,
        irregular: bool | None = None,
        stochastic_level: bool = False,
        stochastic_trend: bool = False,
        stochastic_seasonal: bool = False,
        stochastic_cycle: bool = False,
        damped_cycle: bool = False,
        cycle_period_bounds: CyclePeriodBoundsParams = CyclePeriodBoundsParams(),
        **kwargs: dict[
            str, bool | int | float | None
        ],  # For future extensibility and ignore unknown parameters
    ):
        self.autoregressive = autoregressive
        self.trend = trend
        self.level = level
        self.seasonal = seasonal
        self.freq_seasonal = freq_seasonal
        self.cycle = cycle
        self.irregular = irregular
        self.stochastic_level = stochastic_level
        self.stochastic_trend = stochastic_trend
        self.stochastic_seasonal = stochastic_seasonal
        self.stochastic_cycle = stochastic_cycle
        self.damped_cycle = damped_cycle
        self.cycle_period_bounds = cycle_period_bounds

    def parse(self):
        param_names = [k for k in self.__dict__ if not k.startswith("_")]
        kwargs = {}
        for p in param_names:
            v = getattr(self, p)
            if isinstance(v, BaseParams):
                parsed_value = v.parse()
                if isinstance(parsed_value, dict):
                    for key, value in parsed_value.items():
                        kwargs[key] = value
                else:
                    kwargs[p] = parsed_value
            else:
                kwargs[p] = v
        return kwargs

    @classmethod
    def from_cli_args(cls, **kwargs):
        """
        Create a StructuralTimeSeriesParams instance from CLI arguments.
        """
        freq_seasonal_strs = kwargs.pop("freq_seasonal", [])
        freq_seasonal = FrequencySeasonalityParamsList.from_strings(freq_seasonal_strs)
        cycle_period_bounds_str = kwargs.pop("cycle_period_bounds", ".5,infinity")
        cycle_period_bounds = CyclePeriodBoundsParams.from_string(
            cycle_period_bounds_str
        )
        return cls(
            freq_seasonal=freq_seasonal,
            cycle_period_bounds=cycle_period_bounds,
            **kwargs,
        )


class Visualizaiton:
    def __init__(
        self,
        observed,
        fitted,
        trend,
        freq_seasonal,
        seasonal,
        autoregressive,
    ) -> None:
        pass
