import typer
from typing_extensions import Annotated

app = typer.Typer(help="Data related commands for StatSpace CLI")


@app.command()
def fetch(
    freq: Annotated[str, typer.Option(help="Frequency of the data")] = "h",
    path: Annotated[
        str | None, typer.Option(help="Path to save the data file.")
    ] = None,
    station: Annotated[
        str,
        typer.Option(
            help="The SMHI station ID to fetch data for.",
            rich_help_panel="Data Fetching",
        ),
    ] = "97510",
    parameter: Annotated[
        str,
        typer.Option(
            help="Parameter ID to fetch data for.", rich_help_panel="Data Fetching"
        ),
    ] = "1",
    period: Annotated[
        str,
        typer.Option(
            help="The period for which to fetch data. Possible values: corrected-archive, latest-day, latest-month, latest-hour",
            rich_help_panel="Data Fetching",
        ),
    ] = "corrected-archive",
    skiprows: Annotated[
        int,
        typer.Option(
            help="Number of rows to skip in the data file.",
            rich_help_panel="Data Fetching",
        ),
    ] = 10,
):
    """
    Fetches data from the SMHI API for a given station.
    """
    from statspace.data import (
        get_smhi_data,
    )

    data = get_smhi_data(
        parameter=int(parameter),
        station=int(station),
        period=period,
        freq=freq,
        skiprows=skiprows,
    )
    if path:
        data.to_csv(path, index=False)
    print(data.head())
