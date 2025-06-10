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
        Annotated[int, typer.Option(help="Length of the cycle component")]
    ] = None,
    cycle_innovation: Annotated[
        bool, typer.Option(help="Order of innovations for the cycle component")
    ] = False,
    cycle_n: Optional[Annotated[int, typer.Option(help="Number of cycles")]] = None,
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
    from .data import read_data, preprocess_data

    if not model_path:
        model_config = (
            StructuralTimeSeriesConfig.from_file(model_config) if model_config else None
        )
        sampler_config = (
            SamplerConfig.from_file(sampler_config) if sampler_config else None
        )
        model = StructuralTimeSeriesBuilder(
            model_config=model_config, sampler_config=sampler_config
        )
    else:
        from .model import load_model

        model = load_model(model_path)
    data = read_data(data_path)
    y = preprocess_data(data, resample="M", test_size=12)[0]
    model.fit(y=y)
    model.save(ouptput_path)


@app.command()
def predict(
    data_path: Annotated[str, typer.Option(help="The path to prediction data")],
    model_config: Annotated[str, typer.Option(help="Path to model configuration file")],
    sampler_config: Optional[
        Annotated[str, typer.Option(help="The path to sampler configuration")]
    ] = None,
):
    model_config = (
        StructuralTimeSeriesConfig.from_file(model_config) if model_config else None
    )
    sampler_config = SamplerConfig.from_file(sampler_config) if sampler_config else None
    model = StructuralTimeSeriesBuilder(
        model_config=model_config, sampler_config=sampler_config
    )
    data = read_data(data_path)
    train, test = preprocess_data(data, resample="M", test_size=12)
    predictions = model.predict(train, test)
    print(predictions)


if __name__ == "__main__":
    app()
