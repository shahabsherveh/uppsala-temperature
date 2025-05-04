import yaml
import argparse

parser = argparse.ArgumentParser(description="Generate model config")
parser.add_argument(
    "--name",
    type=str,
)

args = parser.parse_args()

seasons = 12
lags = 1
priors = {
    "ar": {
        "coeffs": {
            "mu": [0 for i in range(lags + 1)],
            "sigma": [1 for i in range(lags + 1)],
            "size": lags + 1,
        },
        "sigma": 2,
        "init": {"mu": 1.2, "sigma": 1, "size": lags + 1},
    },
    "trend": {
        "slope": {"sigma": 1, "init": {"mu": 0, "sigma": 2}},
        "sigma": 2,
        "init": {"mu": 0, "sigma": 1},
    },
    "seasonality": {"sigma": 2, "init": {"mu": 0, "sigma": 1, "size": seasons - 1}},
    "sigma": 2,
}

with open(f"models/{args.name}.yaml", "w") as f:
    yaml.dump(priors, f)
