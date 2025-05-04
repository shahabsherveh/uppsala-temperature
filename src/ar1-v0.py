#!/usr/bin/env python


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

test_size_years = 10
sample_vis_slice = np.arange(-12 * test_size_years, 0, 1)

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
fig.savefig("../figures/uppsala_monthly_temp.png")

# Set up a dictionary for the specification of our priors
# We set up the dictionary to specify size of the AR coefficients in
# case we want to vary the AR lags.
lags = 2
priors = {
    "coefs": {
        "mu": [0 for i in range(lags)],
        "sigma": [1 for i in range(lags)],
        "size": lags,
    },
    "sigma": 2,
    "init": {"mu": 5, "sigma": 2, "size": 1},
}
train = monthly_temp.iloc[: -12 * test_size_years].copy()
test = monthly_temp.iloc[-12 * test_size_years :].copy()
# Initialise the model
with pm.Model() as AR:
    pass

# Define the time interval for fitting the data
t_data_train = np.arange(train.index.shape[0])
t_data_test = np.arange(
    train.index.shape[0], train.index.shape[0] + test.index.shape[0]
)
# Add the time interval as a mutable coordinate to the model to allow for
# future predictions
AR.add_coord("obs_id", t_data_train)

with AR:
    # Data containers to enable prediction
    t = pm.Data("t", t_data_train, dims="obs_id")
    y = pm.Data("y", train, dims="obs_id")

    # The first coefficient will be the constant term but we need to set priors
    # for each coefficient in the AR process
    coefs = pm.Normal("coefs", priors["coefs"]["mu"], priors["coefs"]["sigma"])
    sigma = pm.HalfNormal("sigma", priors["sigma"])
    # We need one init variable for each lag, hence size is variable too
    init = pm.Normal.dist(
        priors["init"]["mu"], priors["init"]["sigma"], size=priors["init"]["size"]
    )
    # Steps of the AR model minus the lags required
    ar2 = pm.AR(
        "ar",
        coefs,
        sigma=sigma,
        init_dist=init,
        constant=True,
        steps=t_data_train.shape[0] - (priors["coefs"]["size"] - 1),
        dims="obs_id",
    )

    # The Likelihood
    outcome = pm.Normal("likelihood", mu=ar2, sigma=sigma, observed=y, dims="obs_id")
    # Sampling
    idata_ar = pm.sample(2000)
    idata_ar.extend(pm.sample_posterior_predictive(idata_ar))


fig, ax = plt.subplots(figsize=(12, 6))
az.plot_ppc(idata_ar, num_pp_samples=100, ax=ax)
fig.suptitle("Posterior Predictive Check")
fig.savefig("../figures/ar1_v0_ppc.png")


summary = az.summary(idata_ar, var_names=["~ar"])
summary.to_csv("../figures/ar1_v0_summary.csv")


fig, ax = plt.subplots(figsize=(10, 4))
idata_ar["posterior"]["ar"].mean(["chain", "draw"])[sample_vis_slice].plot(
    ax=ax, label="Posterior Mean AR level"
)
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
fig.savefig("../figures/ar1_v0_fit.png")


with AR:
    # We need to have coords for the observations minus the lagged term to correctly centre the prediction step
    AR.add_coords(
        {
            "obs_id_fut_1": np.concatenate([t_data_train[-1:], t_data_test]).tolist(),
            "obs_id_fut": t_data_test,
        }
    )
    # condition on the learned values of the AR process
    # initialise the future AR process precisely at the last observed value in the AR process
    # using the special feature of the dirac delta distribution to be 0 everywhere else.
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist=pm.DiracDelta.dist(ar2[..., -1]),
        rho=coefs,
        sigma=sigma,
        constant=True,
        dims="obs_id_fut_1",
    )
    yhat_fut = pm.Normal("yhat_fut", mu=ar1_fut[1:], sigma=sigma, dims="obs_id_fut")
    # use the updated values and predict outcomes and probabilities:
    idata_preds = pm.sample_posterior_predictive(
        idata_ar,
        var_names=["likelihood", "yhat_fut"],
        predictions=True,
        random_seed=100,
    )


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

fig.savefig("../figures/ar1_v0_pred.png")
