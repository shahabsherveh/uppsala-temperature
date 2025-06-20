#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import arviz as az
import argparse
import os

parser = argparse.ArgumentParser(description="Produce Model Stats")

parser.add_argument(
    "--model-output",
    type=str,
    help="Path to the model output files",
    default="../outputs/ar1/",
)
args = parser.parse_args()

stats_dir = f"{args.model_output}/stats/"
os.makedirs(stats_dir, exist_ok=True)

test_size_years = 10
sample_vis_slice = np.arange(-12 * test_size_years, 0, 1)

idata_trace = az.from_json(f"{args.model_output}/idata_trace.json")
idata_preds = az.from_json(f"{args.model_output}/idata_preds.json")
n_samples = idata_trace["posterior"]["ar"].shape[0]


df = pd.read_csv(
    "../data/uppsala_tm_1722-2022/uppsala_tm_1722-2022.dat",
    sep=r"\s+",
)
df.columns = ["year", "month", "day", "temp", "temp_corrected", "data_source"]
monthly_temp = df.groupby(["year", "month"])["temp"].mean()
vis_sample = monthly_temp.iloc[sample_vis_slice]
train = monthly_temp.iloc[: -12 * test_size_years].copy()
test = monthly_temp.iloc[-12 * test_size_years :].copy()

# Define the time interval for fitting the data
t_data = np.arange(monthly_temp.index.shape[0])
t_data_train = t_data[: train.index.shape[0]]
t_data_test = t_data[train.index.shape[0] :]


fig, ax = plt.subplots(figsize=(12, 6))
az.plot_ppc(idata_trace, num_pp_samples=min(n_samples, 100), ax=ax)
fig.suptitle("Posterior Predictive Check")
fig.savefig(f"{stats_dir}/ppc.png")


summary = az.summary(idata_trace, var_names=["~ar"])
summary.to_csv(f"{stats_dir}/summary.csv")


fig, ax = plt.subplots(figsize=(10, 4))
idata_trace["posterior_predictive"].likelihood.mean(["chain", "draw"])[
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
fig.savefig(f"{stats_dir}/fit.png")

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

fig.savefig(f"{stats_dir}/pred.png")
