import typer
from typing import Annotated
from datetime import datetime
from copy import deepcopy


app = typer.Typer()


@app.command()
def configure(
    autoregressive: Annotated[
        int, typer.Option(help="AR order", rich_help_panel="Model Parameters")
    ] = 1,
    trend: Annotated[
        bool,
        typer.Option(
            help="Whether to include a trend component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    level: Annotated[
        bool,
        typer.Option(
            help="Whether to include a level component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    seasonal: Annotated[
        int,
        typer.Option(help="Seasonal length", rich_help_panel="Model Parameters"),
    ] = None,
    freq_seasonal: Annotated[
        list[str],
        typer.Option(
            help="a comma separated key=value pairs of the form 'period=<period>,harmonics=<haramonics>,stochastic=<stochastic>' where <period> and <harmonics> are integers and <stochastic> is a boolean value. This will create a frequency seasonal component with the specified period and harmonics.",
            rich_help_panel="Model Parameters",
        ),
    ] = [],
    cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to include a cycle component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    irregular: Annotated[
        bool,
        typer.Option(
            help="Whether to include an irregular component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_level: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic level",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_trend: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic trend",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_seasonal: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic seasonal",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic cycle",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    damped_cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to use damped cycle", rich_help_panel="Model Parameters"
        ),
    ] = False,
    cycle_period_bounds: Annotated[
        str,
        typer.Option(
            help="A comma seperated values of lower and upper allowed bounds for the period of the cycle.",
            rich_help_panel="Model Parameters",
        ),
    ] = "lower=.5,upper=infinity",
):
    cli_args = locals()
    from statspace.models import StructuralTimeSeriesParams
    import json

    config = StructuralTimeSeriesParams.from_cli_args(**cli_args)
    print(json.dumps(config.parse(), indent=4))
    return config


@app.command()
def train(
    dataset_path: Annotated[
        str,
        typer.Argument(
            help="Path to the dataset file", rich_help_panel="Dataset Parameters"
        ),
    ],
    endog_var: Annotated[
        str,
        typer.Argument(
            help="Endogenous variable name", rich_help_panel="Dataset Parameters"
        ),
    ],
    exog_var: Annotated[
        list[str],
        typer.Option(
            help="Exogenous variable names", rich_help_panel="Dataset Parameters"
        ),
    ] = [],
    timestamp_var: Annotated[
        str | None,
        typer.Option(
            help="Timestamp column name", rich_help_panel="Dataset Parameters"
        ),
    ] = None,
    training_start_date: Annotated[
        datetime | None,
        typer.Option(help="Training ", rich_help_panel="Dataset Parameters"),
    ] = None,
    training_end_date: Annotated[
        datetime | None,
        typer.Option(help="Training end date", rich_help_panel="Dataset Parameters"),
    ] = None,
    resample_freq: Annotated[
        str,
        typer.Option(help="Resampling frequency", rich_help_panel="Dataset Parameters"),
    ] = "h",
    agg_function: Annotated[
        str,
        typer.Option(
            help="Aggregation function for resampling",
            rich_help_panel="Dataset Parameters",
        ),
    ] = "mean",
    forecast_horizon: Annotated[
        int | None,
        typer.Option(help="Forecast horizon", rich_help_panel="Forecast Parameters"),
    ] = None,
    autoregressive: Annotated[
        int, typer.Option(help="AR order", rich_help_panel="Model Parameters")
    ] = 1,
    trend: Annotated[
        bool,
        typer.Option(
            help="Whether to include a trend component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    level: Annotated[
        bool,
        typer.Option(
            help="Whether to include a level component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    seasonal: Annotated[
        int | None,
        typer.Option(help="Seasonal length", rich_help_panel="Model Parameters"),
    ] = None,
    freq_seasonal: Annotated[
        list[str],
        typer.Option(
            help="a comma separated key=value pairs of the form 'period=<period>,harmonics=<haramonics>,stochastic=<stochastic>' where <period> and <harmonics> are integers and <stochastic> could be 'true' or 'false'. It will create a frequency seasonal component with the specified period and harmonics.",
            rich_help_panel="Model Parameters",
        ),
    ] = [],
    cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to include a cycle component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    irregular: Annotated[
        bool,
        typer.Option(
            help="Whether to include an irregular component",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_level: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic level",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_trend: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic trend",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_seasonal: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic seasonal",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    stochastic_cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to use stochastic cycle",
            rich_help_panel="Model Parameters",
        ),
    ] = False,
    damped_cycle: Annotated[
        bool,
        typer.Option(
            help="Whether to use damped cycle", rich_help_panel="Model Parameters"
        ),
    ] = False,
    cycle_period_bounds: Annotated[
        str,
        typer.Option(
            help="A comma seperated values of lower and upper allowed bounds for the period of the cycle.",
            rich_help_panel="Model Parameters",
        ),
    ] = "lower=.5,upper=infinity",
):
    cli_args = deepcopy(locals())
    from statspace.models.structural import StructuralTimeSeriesParams, Visualization
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    import pandas as pd
    from rich import print

    df = pd.read_csv(
        dataset_path,
        parse_dates=[timestamp_var] if timestamp_var else None,
        index_col=timestamp_var if timestamp_var else None,
    )[exog_var + [endog_var]]
    df_resampled = df.resample(resample_freq).agg(agg_function)
    df_resampled = (
        df_resampled[training_start_date:] if training_start_date else df_resampled
    )

    train = df_resampled[:training_end_date] if training_end_date else df_resampled
    test = df_resampled[train.shape[0] : train.shape[0] + forecast_horizon]

    exog_train = train[exog_var] if exog_var else None
    exog_test = test[exog_var] if exog_var else None
    endog_train = train[endog_var]
    endog_test = test[endog_var]

    model_config = StructuralTimeSeriesParams.from_cli_args(**cli_args)
    model = UnobservedComponents(
        endog=endog_train, exog=exog_train, **model_config.parse()
    )
    results = model.fit()

    # Create DataFrames for plotting
    prediction = results.get_prediction()
    fitted = prediction.predicted_mean
    ci = prediction.conf_int()

    # Components
    fitted_trend = results.trend["smoothed"] if results.trend else None
    fitted_level = results.level["smoothed"] if results.level else None
    fitted_seasonal = results.seasonal["smoothed"] if results.seasonal else None
    fitted_freq_seasonal = (
        [s["smoothed"] for s in results.freq_seasonal]
        if results.freq_seasonal
        else None
    )
    fitted_cycle = results.cycle["smoothed"] if results.cycle else None
    fitted_autoregressive = (
        results.autoregressive["smoothed"] if results.autoregressive else None
    )

    # Combine into a dictionary for easy access
    # viz_data = {
    #     "observed": endog_train,
    #     "fitted": fitted,
    #     "ci": ci,
    #     "trend": fitted_trend,
    #     "level": fitted_level,
    #     "freq_seasonal": fitted_freq_seasonal,
    #     "cycle": fitted_cycle,
    #     "autoregressive": fitted_autoregressive,
    #     "seasonal": fitted_seasonal,
    #     "residuals": results.resid,
    #     "dates": endog_train.index,
    # }
    # if forecast_horizon:
    #     forecast = results.get_forecast(steps=forecast_horizon, exog=exog_test)
    #     print(forecast.summary_frame())
    #     # Forecast data
    #     viz_data["forecast_dates"] = forecast.predicted_mean.index
    #     viz_data["forecast_actual"] = endog_test
    #     viz_data["forecast_mean"] = forecast.predicted_mean
    #     viz_data["forecast_ci"] = forecast.conf_int()
    #     viz_data["forecast_resids"] = forecast.predicted_mean - endog_test

    viz = Visualization(
        observed=endog_train,
        model=model,
        results=results,
        unobserved_future_data=endog_test,
    )
    print("Fitted Model Summary:")
    print(results.summary())
    viz.run()


if __name__ == "__main__":
    app()
