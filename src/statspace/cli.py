import typer
from typing import Optional, Annotated

app = typer.Typer()


@app.command()
def create_model_config(
    trend_order: Optional[
        Annotated[int, typer.Option(help="Order of the trend component")]
    ] = None,
    trend_innovations_order: Annotated[
        int, typer.Option(help="Order of innovations for the trend component")
    ] = 0,
    ar_order: Optional[
        Annotated[int, typer.Option(help="Order of the AR component")]
    ] = 1,
    season_length: Optional[
        Annotated[int, typer.Option(help="Length of the seasonal component")]
    ] = None,
    season_innovation: Annotated[
        bool, typer.Option(help="Order of innovations for the seasonal component")
    ] = False,
    seasonal_name: Annotated[
        str, typer.Option(help="Name of the seasonal component")
    ] = "annual",
    cycle_length: Optional[
        Annotated[list[int], typer.Option(help="Length of the cycle component")]
    ] = None,
    cycle_innovation: Annotated[
        list[bool],
        typer.Option(help="Order of innovations for the cycle component"),
    ] = [False],
    cycle_n: Optional[
        Annotated[list[int], typer.Option(help="Number of cycles")]
    ] = None,
):
    from .model import StructuralTimeSeriesConfig

    config = StructuralTimeSeriesConfig(
        trend_order=trend_order,
        trend_innovations_order=trend_innovations_order,
        ar_order=ar_order,
        season_length=season_length,
        season_innovation=season_innovation,
        seasonal_name=seasonal_name,
        cycle_length=cycle_length,
        cycle_innovation=cycle_innovation,
        cycle_n=cycle_n,
    )
    config.to_file("model_config.json")
    print("Model configuration file created: model_config.json")


@app.command()
def create_sampler_config(
    n_chains: Annotated[int, typer.Option(help="Number of chains")] = 4,
    n_samples: Annotated[int, typer.Option(help="Number of samples per chain")] = 1000,
    n_burn: Annotated[int, typer.Option(help="Number of burn-in samples")] = 500,
    step_scale: Annotated[
        float, typer.Option(help="Step scale for the sampler")
    ] = 0.25,
    target_accept: Annotated[
        float, typer.Option(help="Target acceptance rate for the sampler")
    ] = 0.8,
    gamma: Annotated[
        float, typer.Option(help="Gamma parameter for the sampler")
    ] = 0.05,
    k: Annotated[float, typer.Option(help="K parameter for the sampler")] = 0.75,
    t0: Annotated[int, typer.Option(help="T0 parameter for the sampler")] = 10,
    adapt_step_size: Annotated[
        bool, typer.Option(help="Whether to adapt the step size")
    ] = True,
):
    from .model import SamplerConfig

    nuts_sampler_kwargs = {
        "step_scale": step_scale,
        "target_accept": target_accept,
        "gamma": gamma,
        "k": k,
        "t0": t0,
        "adapt_step_size": adapt_step_size,
    }

    config = SamplerConfig(
        n_chains=n_chains,
        n_samples=n_samples,
        n_burn=n_burn,
        nuts_sampler_kwargs=nuts_sampler_kwargs,
    )
    config.to_file("sampler_config.json")
    print("Sampler configuration file created: sampler_config.json")


@app.command()
def train(
    data_path: Annotated[str, typer.Option(help="The path to training data")],
    model_config_path: Optional[
        Annotated[str, typer.Option(help="Path to model configuration file")]
    ] = None,
    sampler_config_path: Optional[
        Annotated[str, typer.Option(help="The path to sampler configuration")]
    ] = None,
    output_path: Optional[
        Annotated[str, typer.Option(help="Path to save the model")]
    ] = None,
    model_path: Optional[
        Annotated[str, typer.Option(help="Path to a pre-trained model")]
    ] = None,
):
    from .model import (
        StructuralTimeSeriesBuilder,
        StructuralTimeSeriesConfig,
        SamplerConfig,
    )
    import pytensor as pt

    pt.config.blas__ldflags = "-lblas -llapack"
    pt.config.allow_gc = False
    from .data import read_data, preprocess_data

    if not model_path:
        model_config = (
            StructuralTimeSeriesConfig.from_file(model_config_path).to_dict()
            if model_config_path
            else None
        )
        sampler_config = (
            SamplerConfig.from_file(sampler_config_path).to_dict()
            if sampler_config_path
            else None
        )
        model = StructuralTimeSeriesBuilder(
            model_config=model_config, sampler_config=sampler_config
        )
    else:
        model = StructuralTimeSeriesBuilder.load(model_path)
    data = read_data(data_path)
    y = preprocess_data(data, resample_freq="ME", start_year="1950", end_year="2020")
    print(f"Training data shape: {y.shape}")
    model.fit(y=y)
    if output_path:
        model.save(output_path)


@app.command()
def forecast(
    model_path: Annotated[str, typer.Option(help="Path to the trained model")],
    steps: Annotated[int, typer.Option(help="Number of steps to forecast")] = 12,
    output_path: Annotated[
        str, typer.Option(help="Path to save the forecast results")
    ] = "predictions.nc",
):
    from .model import (
        StructuralTimeSeriesBuilder,
        SamplerConfig,
    )
    import pytensor as pt

    pt.config.blas__ldflags = "-lblas -llapack"
    pt.config.allow_gc = False

    model = StructuralTimeSeriesBuilder.load(model_path)
    predictions = model.forecast(steps=steps)
    predictions.to_netcdf(output_path)


if __name__ == "__main__":
    app()
