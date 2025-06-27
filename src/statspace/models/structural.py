import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import (
    UnobservedComponentsResults,
    UnobservedComponents,
)
from .utils import BaseParams

import dash
from dash import (
    Dash,
    html,
    dcc,
    callback,
    dash_table,
)
from dash.dependencies import Input, Output
import plotly.express as px


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


class Visualization:
    def __init__(
        self,
        observed,
        model: UnobservedComponents,
        results: UnobservedComponentsResults,
        unobserved_future_data: pd.DataFrame,
        tab_font_size: int = 24,
        graph_title_font_size: int = 18,
        tables_font_size: int = 14,
    ) -> None:
        self.observed = observed
        self.model = model
        self.results = results
        self.unobserved_future_data = unobserved_future_data
        self.tab_font_size = tab_font_size
        self.graph_title_font_size = graph_title_font_size
        self.tables_font_size = tables_font_size
        self.dash_layout = None
        self.trimmed_residuals = self.trim_residuals()
        self.prediction_graphs = {
            "actual_vs_fitted": None,
            "residuals": None,
            "residuals_acf": None,
            "residuals_histogram": None,
            "residuals_qq": None,
            "actual_vs_fitted_future": None,
        }
        self.components_graphs = {
            "seasonal_component": None,
            "frequency_seasonal_components": None,
            "cycle_component": None,
            "irregular_component": None,
            "trend_component": None,
            "level_component": None,
            "autoregressive_component": None,
        }

        self.model_diagnostics_graphs = {
            "parameters_statistics": None,
            "model_summary": None,
            "model_tests": None,
        }
        self.app = Dash(__name__)

    def trim_residuals(self):
        """
        Trim the initial points of the residuals based on the maximum initial points
        required by the model components.
        """
        max_initial_points = max(
            [self.model.loglikelihood_burn, self.results.nobs_diffuse]
        )

        return self.results.resid[max_initial_points:]

    def build_layout(self):
        self.dash_layout = html.Div(
            [
                html.H1("Structural Time Series Model Dashboard"),
                dcc.Tabs(
                    [
                        dcc.Tab(
                            label="Prediction",
                            children=[
                                g
                                for g in self.prediction_graphs.values()
                                if g is not None
                            ],
                        ),
                        dcc.Tab(
                            label="Components",
                            children=[
                                g
                                for g in self.components_graphs.values()
                                if g is not None
                            ],
                        ),
                        dcc.Tab(
                            label="Diagnostics",
                            children=[
                                g
                                for g in self.model_diagnostics_graphs.values()
                                if g is not None
                            ],
                        ),
                    ],
                    style=dict(fontSize=self.tab_font_size),
                ),
            ]
        )

    def build_app(self):
        self.app = Dash(__name__)
        self.build_prediction_graphs()
        self.build_components_graphs()
        self.build_model_diagnostics_graphs()
        self.build_layout()
        self.app.layout = self.dash_layout

    def build_prediction_graphs(self):
        for key in self.prediction_graphs:
            if hasattr(self, f"plot_{key}"):
                figure = getattr(self, f"plot_{key}")()
                if figure:
                    figure.update_layout(font_size=self.graph_title_font_size)
                    self.prediction_graphs[key] = dcc.Graph(
                        figure=figure,
                        id=key,
                        style={
                            "display": "inline-block",
                            "width": "50%",
                            "height": "800px",
                        },
                    )

    def build_components_graphs(self):
        for key in self.components_graphs:
            if hasattr(self, f"plot_{key}"):
                figure = getattr(self, f"plot_{key}")()
                if figure:
                    figure.update_layout(font_size=self.graph_title_font_size)
                    self.components_graphs[key] = dcc.Graph(
                        figure=figure,
                        id=key,
                        style={
                            "display": "inline-block",
                            "width": "50%",
                            "height": "800px",
                        },
                    )

    def build_model_diagnostics_graphs(self):
        for key in self.model_diagnostics_graphs:
            if hasattr(self, f"plot_{key}"):
                figure = getattr(self, f"plot_{key}")()
                if figure:
                    self.model_diagnostics_graphs[key] = dcc.Graph(
                        figure=figure,
                        id=key,
                        style={
                            "display": "inline-block",
                            "width": "100%",
                        },
                    )

    def run(self, **kwargs):
        self.build_app()
        self.app.run(**kwargs)

    # Example plot methods (implement these as needed)
    def plot_actual_vs_fitted(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.observed.index, y=self.observed, mode="lines", name="Actual"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.observed.index,
                y=self.results.fittedvalues,
                mode="lines",
                name="Fitted",
            )
        )
        fig.update_layout(title="Actual vs Fitted")
        return fig

    def plot_seasonal_component(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "seasonal"):
            fig.add_trace(
                go.Scatter(
                    x=self.observed.index,
                    y=self.results.seasonal["smoothed"],
                    mode="lines",
                    name=f"Seasonal {self.model.seasonal_periods}",
                )
            )
            fig.update_layout(title="Seasonal Component")
            return fig

    # Implement the rest of the plot_* methods similarly
    def plot_frequency_seasonal_components(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "freq_seasonal"):
            for i, fs in enumerate(self.results.freq_seasonal):
                fig.add_trace(
                    go.Scatter(
                        x=self.observed.index,
                        y=fs["smoothed"],
                        mode="lines",
                        name=f"Periods: {self.model.freq_seasonal_periods[i]} Harmonics: {self.model.freq_seasonal_harmonics[i]} Stochastic: {self.model.stochastic_freq_seasonal[i]}",
                    )
                )
            fig.update_layout(title="Frequency Seasonal Component")
            return fig

    def plot_cycle_component(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "cycle"):
            fig.add_trace(
                go.Scatter(
                    x=self.observed.index,
                    y=self.results.cycle["smoothed"],
                    mode="lines",
                    name="Cycle Component",
                )
            )
            fig.update_layout(title="Cycle Component")
            return fig

    def plot_trend_component(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "trend"):
            fig.add_trace(
                go.Scatter(
                    x=self.observed.index,
                    y=self.results.trend["smoothed"],
                    mode="lines",
                    name="Trend Component",
                )
            )
            fig.update_layout(title="Trend Component")
            return fig

    def plot_level_component(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "level"):
            fig.add_trace(
                go.Scatter(
                    x=self.observed.index,
                    y=self.results.level["smoothed"],
                    mode="lines",
                    name="Level Component",
                )
            )
            fig.update_layout(title="Level Component")
            return fig

    def plot_autoregressive_component(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        if getattr(self.results, "autoregressive"):
            fig.add_trace(
                go.Scatter(
                    x=self.observed.index,
                    y=self.results.autoregressive["smoothed"],
                    mode="lines",
                    name="Autoregressive Component",
                )
            )
            fig.update_layout(title="Autoregressive Component")
            return fig

    def plot_residuals(self):
        import plotly.graph_objs as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.observed.index,
                y=self.results.resid,
                mode="lines",
                name="Residuals",
            )
        )
        fig.update_layout(title="Residuals")
        return fig

    def plot_residuals_acf(self):
        import plotly.graph_objs as go
        from statsmodels.tsa import stattools

        acf_interval = stattools.acf(self.results.resid, nlags=40, alpha=0.05)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(acf_interval[0])),
                y=acf_interval[0],
                mode="lines+markers",
                name="ACF",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [
                        np.arange(len(acf_interval[0])),
                        np.arange(len(acf_interval[0]))[::-1],
                    ]
                ),
                y=np.concatenate([acf_interval[1][:, 1], acf_interval[1][::-1, 0]]),
                mode="lines",
                name="95% Confidence Interval",
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.update_layout(
            title="ACF of Residuals", xaxis_title="Lag", yaxis_title="ACF"
        )
        return fig

    def plot_residuals_histogram(self):
        from plotly.figure_factory import create_distplot

        normal_data = np.random.normal(scale=self.results.resid.std(), size=10000)

        fig = create_distplot(
            [self.trimmed_residuals, normal_data],
            group_labels=["Residuals", "N(0,1)"],
            bin_size=0.5,
            colors=["blue", "orange"],
        )
        fig.add_vline(0, line_width=2, line_color="orange")
        fig.add_vline(self.results.resid.mean(), line_width=2, line_color="blue")
        fig.update_layout(title="Residuals Histogram", xaxis_title="Residuals")
        return fig

    def plot_residuals_qq(self):
        import plotly.graph_objs as go
        from statsmodels.graphics.gofplots import qqplot

        qqplot_data = qqplot(self.trimmed_residuals, line="s", fit=True).gca().lines
        fig = go.Figure()
        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[0].get_xdata(),
                "y": qqplot_data[0].get_ydata(),
                "mode": "markers",
                "marker": {"color": "#19d3f3"},
            }
        )

        fig.add_trace(
            {
                "type": "scatter",
                "x": qqplot_data[1].get_xdata(),
                "y": qqplot_data[1].get_ydata(),
                "mode": "lines",
                "line": {"color": "#636efa"},
            }
        )

        fig["layout"].update(
            {
                "title": "Quantile-Quantile Plot",
                "xaxis": {"title": "Theoritical Quantities", "zeroline": False},
                "yaxis": {"title": "Sample Quantities"},
                "showlegend": False,
                "width": 800,
                "height": 700,
            }
        )
        fig.update_layout(title="Q-Q Plot of Residuals")
        return fig

    def plot_actual_vs_fitted_future(self):
        import plotly.graph_objs as go

        prediction = self.results.get_prediction(
            start=self.unobserved_future_data.index[0],
            end=self.unobserved_future_data.index[-1],
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.unobserved_future_data.index,
                y=self.unobserved_future_data,
                mode="lines",
                name="Actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.unobserved_future_data.index,
                y=prediction.predicted_mean,
                mode="lines",
                name="Fitted Future",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.unobserved_future_data.index,
                y=self.observed[
                    self.unobserved_future_data.index - pd.Timedelta(365, unit="D")
                ],
                mode="lines",
                name="Previous Year",
                line=dict(dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.unobserved_future_data.index,
                y=self.observed[
                    self.unobserved_future_data.index - pd.Timedelta(2 * 365, unit="D")
                ],
                mode="lines",
                name="Two Years Ago",
                line=dict(dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [
                        self.unobserved_future_data.index,
                        self.unobserved_future_data.index[::-1],
                    ]
                ),
                y=np.concatenate(
                    [
                        prediction.conf_int().iloc[:, 0],
                        prediction.conf_int().iloc[::-1, 1],
                    ]
                ),
                mode="lines",
                name="95% Confidence Interval",
                fill="toself",
                fillcolor="rgba(0,100,80,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
            )
        )
        fig.update_layout(title="Actual vs Fitted Future")
        return fig

    def plot_parameters_statistics(self):
        import plotly.graph_objs as go

        summary = self.results.summary().tables[1]
        df_summary = pd.DataFrame(summary.data[1:], columns=summary.data[0])
        fig = go.Figure()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=df_summary.columns.tolist(),
                    fill_color="paleturquoise",
                    align="left",
                    font_size=self.tables_font_size,
                    height=40,
                ),
                cells=dict(
                    values=[df_summary[col] for col in df_summary.columns],
                    fill_color="lavender",
                    align="left",
                    font_size=self.tables_font_size,
                    height=30,
                ),
            )
        )

        fig.update_layout(
            title=dict(
                text="Parameters Statistics", font_size=self.graph_title_font_size
            )
        )
        return fig

    def plot_model_summary(self):
        import plotly.graph_objs as go
        from io import StringIO

        summary = self.results.summary().tables[0]
        summary_html = summary.as_html()
        df = pd.read_html(StringIO(summary_html))[0]
        df.replace(pd.NA, "", inplace=True)
        df.replace(np.nan, "", inplace=True)
        fig = go.Figure()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[""]
                    * df.shape[1],  # num_columns = number of columns in your table
                    fill=dict(color="rgba(0,0,0,0)"),
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                    font_size=self.tables_font_size,
                    height=30,
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
            )
        )
        fig.update_layout(
            title=dict(text="Model Summary", font_size=self.graph_title_font_size)
        )
        return fig

    def plot_model_tests(self):
        import plotly.graph_objs as go
        from io import StringIO

        summary = self.results.summary().tables[2]
        summary_html = summary.as_html()
        df = pd.read_html(StringIO(summary_html))[0]
        fig = go.Figure()
        fig.add_trace(
            go.Table(
                header=dict(
                    values=[""]
                    * df.shape[1],  # num_columns = number of columns in your table
                    fill=dict(color="rgba(0,0,0,0)"),
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="lavender",
                    align="left",
                    font_size=self.tables_font_size,
                    height=30,
                ),
            )
        )
        fig.update_layout(
            title=dict(text="Model Tests", font_size=self.graph_title_font_size)
        )
        return fig
