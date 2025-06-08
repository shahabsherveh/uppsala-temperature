import typer
from typing import Optional, Annotated

app = typer.Typer()


@app.command()
def train(
    data_path: Annotated[str, typer.Option(help="The path to training data")],
    model_config: Optional[
        Annotated[str, typer.Option(help="Path to model configuration file")]
    ] = None,
    sampler_config: Optional[
        Annotated[str, typer.Option(help="The path to sampler configuration")]
    ] = None,
    ouptput_path: Annotated[
        str, typer.Option(help="Path to save the model")
    ] = "model.nc",
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
