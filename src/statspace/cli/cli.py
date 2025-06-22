import typer

from .structural import app as structural_app
from .data import app as data_app

app = typer.Typer()
app.add_typer(structural_app, name="structural", help="Structural time series commands")
app.add_typer(data_app, name="data", help="SMHI Data fetching commands")


if __name__ == "__main__":
    app()
