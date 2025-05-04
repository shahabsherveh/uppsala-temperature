#!/usr/bin/env python


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pathlib
import os


test_size_years = 10
sample_vis_slice = np.arange(-12 * test_size_years, 0, 1)
model_name = pathlib.Path(__file__).stem
figure_dir = f"../figures/{model_name}/"
debug = True

if debug:
    n_samples = 10
    n_tune = 0
    n_chain = 1
else:
    n_samples = 1000
    n_tune = 1000
    n_chain = 4

os.makedirs(figure_dir, exist_ok=True)


df = pd.read_csv(
    "../data/uppsala_tm_1722-2022/uppsala_tm_1722-2022.dat",
    sep=r"\s+",
)
df.columns = ["year", "month", "day", "temp", "temp_corrected", "data_source"]


monthly_temp = df.groupby(["year", "month"])["temp"].mean()
vis_sample = monthly_temp.iloc[sample_vis_slice]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(vis_sample.values, label="Actual")
ax.plot(vis_sample.rolling(12).mean().values, label="12 Month Rolling Mean")
ax.legend()
fig.suptitle("Uppsala Monthly Mean Temperature")
fig.savefig(f"{figure_dir}/uppsala_monthly_temp.png")

# Set up a dictionary for the specification of our priors
# We set up the dictionary to specify size of the AR coefficients in
# case we want to vary the AR lags.
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

    # The first coefficient will be the constant term but we need to set priors for each coefficient in the AR process
    ar_prior = priors.get("ar")
    trend_prior = priors.get("trend")
    seasonality_prior = priors.get("seasonality")
    sigma = pm.HalfNormal("sigma", priors["sigma"])
    if ar_prior:
        # We need one init variable for each lag, hence size is variable too
        init_ar = pm.Normal.dist(
            ar_prior["init"]["mu"],
            ar_prior["init"]["sigma"],
            size=ar_prior["init"]["size"] - 1,
        )
        # Steps of the AR model minus the lags required
        sigma_ar = pm.HalfNormal("sigma_ar", ar_prior["sigma"])
        ar_coeffs = pm.Normal(
            "ar_coeffs", ar_prior["coeffs"]["mu"], ar_prior["coeffs"]["sigma"]
        )
        ar = pm.AR(
            "ar",
            ar_coeffs,
            sigma=sigma_ar,
            init_dist=init_ar,
            constant=True,
            steps=t.shape[0] - (ar_prior["coeffs"]["size"] - 1),
            dims="obs_id",
        )
    else:
        ar = 0
    if trend_prior:
        # The trend is a linear function of time
        slope_init = pm.Normal.dist(
            mu=trend_prior["slope"]["init"]["mu"],
            sigma=trend_prior["slope"]["init"]["sigma"],
        )
        slope = pm.GaussianRandomWalk(
            "trend_slope",
            mu=0,
            sigma=trend_prior["slope"]["sigma"],
            init_dist=slope_init,
            steps=t.shape[0] - 2,
        )
        trend_init = pm.Normal.dist(
            mu=trend_prior["init"]["mu"], sigma=trend_prior["init"]["sigma"]
        )
        sigma_trend = pm.HalfNormal("sigma_trend", trend_prior["sigma"])
        trend = pm.GaussianRandomWalk(
            "trend",
            mu=slope,
            sigma=sigma_trend,
            init_dist=trend_init,
            steps=t.shape[0] - 1,
            dims="obs_id",
        )
    else:
        trend = 0

    if seasonality_prior:
        sigma_seasonality = pm.HalfNormal(
            "sigma_seasonality", seasonality_prior["sigma"]
        )
        init_seasonality = pm.Normal.dist(
            mu=seasonality_prior["init"]["mu"],
            sigma=seasonality_prior["init"]["sigma"],
            size=seasonality_prior["init"]["size"],
        )
        seasonality_coeffs = np.full(
            shape=seasonality_prior["init"]["size"], fill_value=-1
        )
        # The Fourier terms are added to the AR process
        # We need to set the size of the Fourier terms to be the number of coefficients
        seasonality = pm.AR(
            "seasonality",
            seasonality_coeffs,
            sigma=sigma_seasonality,
            init_dist=init_seasonality,
            constant=False,
            steps=t.shape[0] - (seasonality_prior["init"]["size"]),
            dims="obs_id",
        )
    else:
        seasonality = 0

    # The Likelihood
    outcome = pm.Normal(
        "likelihood",
        mu=ar + trend + seasonality,
        sigma=sigma,
        observed=y,
        dims="obs_id",
    )
    ## Sampling
    # idata_ar = pm.sample_prior_predictive(draws=10)
    idata_ar = pm.sample(n_samples, tune=n_tune, chains=n_chain)
    idata_ar.extend(pm.sample_posterior_predictive(idata_ar))

idata_ar.to_json(figure_dir + "idata_ar.json")


fig, ax = plt.subplots(figsize=(12, 6))
az.plot_ppc(idata_ar, num_pp_samples=min(n_samples, 100), ax=ax)
fig.suptitle("Posterior Predictive Check")
fig.savefig(f"{figure_dir}/ppc.png")


summary = az.summary(idata_ar, var_names=["~ar"])
summary.to_csv(f"{figure_dir}/summary.csv")


fig, ax = plt.subplots(figsize=(10, 4))
idata_ar["posterior_predictive"].likelihood.mean(["chain", "draw"])[
    sample_vis_slice
].plot(ax=ax, label="Posterior Mean AR level")
ax.plot(
    t_data_train[sample_vis_slice],
    train.values[sample_vis_slice],
    "o",
    color="black",
    markersize=2,
    label="Observed Data",
)
ax.legend()
ax.set_title("Fitted AR process\nand observed data")
fig.savefig(f"{figure_dir}/fit.png")


with AR:
    ## We need to have coords for the observations minus the lagged term to correctly centre the prediction step
    AR.add_coord("obs_id_fut_ar", np.concatenate([t_data_train[-lags:], t_data_test]))
    AR.add_coord("obs_id_fut_trend", np.concatenate([t_data_train[-1:], t_data_test]))
    AR.add_coord(
        "obs_id_fut_seasonality",
        np.concatenate([t_data_train[-seasons + 1 :], t_data_test]),
    )
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
    init_seasonality_fut = pm.DiracDelta.dist(seasonality[..., -seasons + 1 :])
    seasonality_fut = pm.AR(
        "seasonality_fut",
        init_dist=init_seasonality_fut,
        rho=seasonality_coeffs,
        sigma=sigma,
        constant=False,
        dims="obs_id_fut_seasonality",
    )
    init_trend_fut = pm.DiracDelta.dist(trend[..., -1:])
    trend_fut = pm.GaussianRandomWalk(
        "trend_fut",
        mu=0,
        sigma=sigma,
        init_dist=init_trend_fut,
        dims="obs_id_fut_trend",
    )
    yhat_fut = pm.Normal(
        "yhat_fut",
        mu=ar_fut[lags:] + trend_fut[1:] + seasonality_fut[seasons - 1 :],
        sigma=sigma,
        dims="obs_id_fut",
    )
    # use the updated values and predict outcomes and probabilities:
    idata_preds = pm.sample_posterior_predictive(
        idata_ar, var_names=["yhat_fut"], predictions=True
    )

idata_preds.to_json(figure_dir + "idata_preds.json")


fig, ax = plt.subplots(figsize=(12, 6))

idata_preds["predictions"].yhat_fut.mean(["chain", "draw"]).plot(color="red")
ax.plot(t_data_test, test.values, label="Observed Data")
percs = np.linspace(51, 99, 100)
colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))

palette = "plasma"
cmap = plt.get_cmap(palette)

for i, p in enumerate(percs[::-1]):
    upper = np.percentile(idata_preds["predictions"].yhat_fut, p, axis=[0, 1])
    lower = np.percentile(idata_preds["predictions"].yhat_fut, 100 - p, axis=[0, 1])
    color_val = colors[i]

    ax.fill_between(
        x=idata_preds["predictions"].coords["obs_id_fut"].data,
        y1=upper.flatten(),
        y2=lower.flatten(),
        color=cmap(color_val),
        alpha=0.3,
    )

fig.savefig(f"{figure_dir}/pred.png")
