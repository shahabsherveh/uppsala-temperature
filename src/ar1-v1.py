#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df = pd.read_csv('../data/uppsala_tm_1722-2022/uppsala_tm_1722-2022.dat', sep=r'\s+', )
df.columns = ['year', 'month', 'day', 'temp', 'temp_corrected', 'data_source']


# In[3]:


df


# In[4]:


monthly_temp = df.groupby(['year', 'month'])['temp'].mean()


# In[5]:


monthly_temp[-5 * 12:].plot(figsize=(12, 6), title='Daily Mean Temperature')


# In[7]:


## Set up a dictionary for the specification of our priors
## We set up the dictionary to specify size of the AR coefficients in
## case we want to vary the AR lags.
lags = 2
priors = {
    "coefs": {"mu": [0 for i in range(lags)], "sigma": [1 for i in range(lags)], "size": lags},
    "sigma": 2,
    "init": {"mu": 5, "sigma": 2, "size": 1},
    "beta_fourier": {"mu": 0, "sigma": 2},
}
train = monthly_temp[:-48].copy()
test = monthly_temp[-48:].copy()
## Initialise the model
with pm.Model() as AR:
    pass

## Define the time interval for fitting the data
t_data = np.arange(monthly_temp.index.shape[0])
t_data_train = t_data[:train.index.shape[0]]
t_data_test = t_data[train.index.shape[0]:]
fourier_order = 12
periods = t_data / 12
fourier_features = pd.DataFrame(
    {
        f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
        for order in range(1, fourier_order + 1)
        for func in ("sin", "cos")
    }
)
fourier_features_train = fourier_features[:train.index.shape[0]].copy()
fourier_features_test = fourier_features[train.index.shape[0]:].copy()

## Add the time interval as a mutable coordinate to the model to allow for future predictions
AR.add_coord("obs_id", t_data_train)
AR.add_coord("fourier_features", fourier_features_train.columns)

with AR:
    ## Data containers to enable prediction
    t = pm.Data("t", t_data_train, dims="obs_id")
    y = pm.Data("y", train, dims="obs_id")

    # The first coefficient will be the constant term but we need to set priors for each coefficient in the AR process
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
        steps=t.shape[0] - (priors["coefs"]["size"] - 1),
        dims="obs_id",
    )
    # The Fourier terms are added to the AR process
    # We need to set the size of the Fourier terms to be the number of coefficients
    fourier_terms = pm.Normal(
        "fourier_terms",
        mu=0,
        sigma=priors["beta_fourier"]["sigma"],
        dims="fourier_features",
    )

    seasonality = pm.Deterministic('seasonality', pm.math.dot(fourier_terms, fourier_features_train.T), dims="obs_id")

    # The Likelihood
    outcome = pm.Normal("likelihood", mu=ar2 + seasonality, sigma=sigma, observed=y, dims="obs_id")
    ## Sampling
    idata_ar = pm.sample_prior_predictive()
    idata_ar.extend(pm.sample(2000))
    idata_ar.extend(pm.sample_posterior_predictive(idata_ar))


# In[8]:


az.plot_ppc(idata_ar, num_pp_samples=100, figsize=(12, 6))


# In[9]:


az.summary(idata_ar, var_names=["~ar"])


# In[10]:


fig, ax = plt.subplots(figsize=(10, 4))
(idata_ar.posterior.seasonality.mean(["chain", "draw"])[-10 * 12:].values + idata_ar.posterior.ar.mean(
    ["chain", "draw"])[-10 * 12:]).plot(ax=ax, label="Posterior Mean AR level")
ax.plot(t_data_train[-10 * 12:], train.values[-10 * 12:], "o", color="black", markersize=2, label="Observed Data")
ax.legend()
ax.set_title("Fitted AR process\nand observed data");


# In[11]:


with AR:
    ## We need to have coords for the observations minus the lagged term to correctly centre the prediction step
    AR.add_coords({"obs_id_fut_1": np.concatenate([t_data_train[-1:], t_data_test])})
    AR.add_coords({"obs_id_fut": t_data_test})
    AR.add_coord("fourier_features_fut", fourier_features_train.columns)
    # condition on the learned values of the AR process
    # initialise the future AR process precisely at the last observed value in the AR process
    # using the special feature of the dirac delta distribution to be 0 everywhere else.
    ar1_fut = pm.AR(
        "ar1_fut",
        init_dist=pm.DiracDelta.dist(ar2[..., -1]),
        rho=coefs,
        sigma=sigma,
        constant=True,
        dims="obs_id_fut_1"
    )
    seasonality = pm.Deterministic(
        "seasonality_fut", pm.math.dot(fourier_terms, fourier_features_test.T), dims="obs_id_fut"
    )
    yhat_fut = pm.Normal("yhat_fut", mu=ar1_fut[1:] + seasonality, sigma=sigma, dims="obs_id_fut")
    # use the updated values and predict outcomes and probabilities:
    idata_preds = pm.sample_posterior_predictive(
        idata_ar, var_names=["likelihood", "yhat_fut"], predictions=True, random_seed=100
    )


# In[12]:


# idata_preds.predictions.yhat_fut[0].to_pandas().T.plot(alpha=0.01, color="y", legend=False)
idata_preds.predictions.yhat_fut.mean(['chain', 'draw']).plot(color="red")
plt.plot(t_data_test, test.values, label="Observed Data")
percs = np.linspace(51, 99, 100)
colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))

palette = "plasma"
cmap = plt.get_cmap(palette)

for i, p in enumerate(percs[::-1]):
    upper = np.percentile(
        idata_preds.predictions.yhat_fut, p, axis=[0, 1]
    )
    lower = np.percentile(
        idata_preds.predictions.yhat_fut, 100 - p, axis=[0, 1]
    )
    color_val = colors[i]

    plt.fill_between(
        x=idata_preds.predictions.coords["obs_id_fut"].data,
        y1=upper.flatten(),
        y2=lower.flatten(),
        color=cmap(color_val),
        alpha=0.3,
    )

