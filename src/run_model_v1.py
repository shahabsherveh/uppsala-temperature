#!/usr/bin/env python


import numpy as np
import pandas as pd
import pymc as pm
import yaml
import os
import argparse


parser = argparse.ArgumentParser(description="Run PyMC model")

parser.add_argument(
    "--model-config",
    type=str,
    default="models/config.yaml",
)

parser.add_argument("--debug", action="store_true")

parser.add_argument(
    "--sample-size",
    type=int,
    default=1000,
    help="Number of samples to draw from the posterior",
)

parser.add_argument(
    "--test-size-years",
    type=int,
    default=10,
    help="Number of years to use for testing",
)

args = parser.parse_args()

with open(args.model_config, "r") as f:
    model_config = yaml.safe_load(f)

ar_prior = model_config["ar"]
trend_prior = model_config.get("trend")
seasonality_prior = model_config.get("seasonality")

test_size_years = args.test_size_years
debug = args.debug

model_name = model_config["name"]
output_dir = f"../outputs/{model_name}/"
lags = ar_prior["size"] - 1
seasons = model_config.get("seasonality", {}).get("size", 0) + 1

if debug:
    n_samples = 10
    n_tune = 0
    n_chain = 1
else:
    n_samples = args.sample_size
    n_tune = 1000
    n_chain = 4

os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(
    "../data/uppsala_tm_1722-2022/uppsala_tm_1722-2022.dat",
    sep=r"\s+",
)
df.columns = ["year", "month", "day", "temp", "temp_corrected", "data_source"]
monthly_temp = df.groupby(["year", "month"])["temp"].mean()

train = monthly_temp.iloc[: -12 * test_size_years].copy()
test = monthly_temp.iloc[-12 * test_size_years :].copy()

# Define the time interval for fitting the data
t_data = np.arange(monthly_temp.index.shape[0])
t_data_train = t_data[: train.index.shape[0]]
t_data_test = t_data[train.index.shape[0] :]

# Initialise the model
with pm.Model() as AR:
    pass

# Add the time interval as a mutable coordinate to the model to allow for
# future predictions
AR.add_coord("obs_id", t_data_train)

with AR:
    ## Data containers to enable prediction
    t = pm.Data("t", t_data_train, dims="obs_id")
    y = pm.Data("y", train, dims="obs_id")
    mu = 0

    # The first coefficient will be the constant term but we need to set priors for each coefficient in the AR process
    sigma = pm.HalfNormal("sigma", model_config["sigma"])
    # We need one init variable for each lag, hence size is variable too
    init_ar = pm.Normal.dist(
        ar_prior["init"]["mu"],
        ar_prior["init"]["sigma"],
        size=ar_prior["size"],
    )
    # Steps of the AR model minus the lags required
    # sigma_ar = pm.HalfNormal("sigma_ar", ar_prior["sigma"])
    ar_coeffs = pm.MvNormal(
        "ar_coeffs",
        ar_prior["coeffs"]["mu"],
        ar_prior["coeffs"]["sigma"] * np.eye(ar_prior["size"]),
        # shape=ar_prior["size"],
    )
    ar = pm.AR(
        "ar",
        ar_coeffs,
        sigma=sigma,
        init_dist=init_ar,
        constant=True,
        steps=t.shape[0] - (ar_prior["size"] - 1),
        dims="obs_id",
    )
    mu += ar
    if trend_prior:
        # The trend is a linear function of time
        drift = pm.Normal(
            "drift",
            mu=trend_prior["drift"]["mu"],
            sigma=trend_prior["drift"]["sigma"],
        )
        trend_init = pm.Normal.dist(
            mu=trend_prior["init"]["mu"], sigma=trend_prior["init"]["sigma"]
        )
        # sigma = pm.HalfNormal("sigma_trend", trend_prior["sigma"])
        trend = pm.GaussianRandomWalk(
            "trend",
            mu=drift,
            sigma=sigma,
            init_dist=trend_init,
            steps=t.shape[0] - 1,
            dims="obs_id",
        )
        mu += trend

    if seasonality_prior:
        # sigma = pm.HalfNormal(
        #     "sigma_seasonality", seasonality_prior["sigma"]
        # )
        init_seasonality = pm.Normal.dist(
            mu=seasonality_prior["init"]["mu"],
            sigma=seasonality_prior["init"]["sigma"],
            size=seasonality_prior["size"],
        )
        seasonality_coeffs = np.full(shape=seasonality_prior["size"], fill_value=-1)
        # The Fourier terms are added to the AR process
        # We need to set the size of the Fourier terms to be the number of coefficients
        seasonality = pm.AR(
            "seasonality",
            seasonality_coeffs,
            sigma=sigma,
            init_dist=init_seasonality,
            constant=False,
            steps=t.shape[0] - (seasonality_prior["size"]),
            dims="obs_id",
        )
        mu += seasonality

    # The Likelihood
    outcome = pm.Normal(
        "likelihood",
        mu=mu,
        sigma=sigma,
        observed=y,
        dims="obs_id",
    )
    ## Sampling
    # idata_ar = pm.sample_prior_predictive(draws=10)
    idata_trace = pm.sample(n_samples, tune=n_tune, chains=n_chain)
    idata_trace.extend(pm.sample_posterior_predictive(idata_trace))

idata_trace.to_json(output_dir + "idata_trace.json")

with AR:
    ## We need to have coords for the observations minus the lagged term to correctly centre the prediction step
    t_fut_ar = np.concatenate([t_data_train[-lags:], t_data_test])
    t_fut_trend = np.concatenate([t_data_train[-1:], t_data_test])
    t_fut_seasonality = np.concatenate([t_data_train[-seasons + 1 :], t_data_test])
    AR.add_coord("obs_id_fut_ar", t_fut_ar)
    AR.add_coord("obs_id_fut_trend", t_fut_trend)
    AR.add_coord("obs_id_fut_seasonality", t_fut_seasonality)
    AR.add_coord("obs_id_fut", t_data_test)
    # condition on the learned values of the AR process
    # initialise the future AR process precisely at the last observed value in the AR process
    # using the special feature of the dirac delta distribution to be 0 everywhere else.


with AR:
    init_ar_fut = pm.DiracDelta.dist(ar[..., -lags:])
    ar_fut = pm.AR(
        "ar_fut",
        init_dist=init_ar_fut,
        rho=ar_coeffs,
        sigma=sigma,
        constant=True,
        dims="obs_id_fut_ar",
    )
    mu = ar_fut[lags:]
    if seasonality_prior:
        init_seasonality_fut = pm.DiracDelta.dist(seasonality[..., -seasons + 1 :])
        seasonality_fut = pm.AR(
            "seasonality_fut",
            init_dist=init_seasonality_fut,
            rho=seasonality_coeffs,
            sigma=sigma,
            constant=False,
            dims="obs_id_fut_seasonality",
        )
        mu += seasonality_fut[seasons - 1 :]

    if trend_prior:
        init_trend_fut = pm.DiracDelta.dist(trend[..., -1:])
        trend_fut = pm.GaussianRandomWalk(
            "trend_fut",
            mu=drift,
            sigma=sigma,
            init_dist=init_trend_fut,
            dims="obs_id_fut_trend",
        )
        mu += trend_fut[1:]
    yhat_fut = pm.Normal(
        "yhat_fut",
        mu=mu,
        sigma=sigma,
        dims="obs_id_fut",
    )
    # use the updated values and predict outcomes and probabilities:
    idata_preds = pm.sample_posterior_predictive(
        idata_trace, var_names=["yhat_fut"], predictions=True
    )

idata_preds.to_json(output_dir + "idata_preds.json")
pm.model_to_graphviz(AR, save=output_dir + "model_graph.png", figsize=(20, 20))
