import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

version = "v6_quant0.5"
outdir = f"../output/{version}/"
summarydir = '../output/{}/summaries/'.format(version, "temperatures")
fairdir = '../output/{}/fair_{}/'.format(version, "temperatures")
plotdir = outdir + "plots/"
if not os.path.exists(plotdir):
    os.makedirs(plotdir)
if not os.path.exists(summarydir):
    os.makedirs(summarydir)
summaryname = f"temperature_summary_{version}.csv"
timesummaryname = f"time_max_summary{version}.csv"
scenariofiles = [
        x for x in os.listdir(fairdir)
        if x.endswith('.csv') and x not in [summaryname, timesummaryname]
]
# Do we plot everything in one place?
single_plot = False
savestring = f"Delayed_temps_{version}"

results = []
quantiles = [0.1, 0.2, 0.3, 0.33, 0.4, 0.5, 0.6, 0.66, 0.7, 0.8, 0.9]
for scen in scenariofiles:
    tmp = pd.read_csv(fairdir + scen, index_col="year")
    for q in quantiles:
        quant_res = tmp.quantile(q, axis=1)
        quant_res = pd.DataFrame(quant_res).T
        quant_res["quantile"] = q
        quant_res["scenario"] = scen[:-4].replace("_Harmonized", "")
        results.append(quant_res)

results = pd.concat(results)
results.to_csv(summarydir + summaryname)

# Design a color scheme
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(set(results["scenario"]))))
scenset = list(set(results["scenario"]))
cdict = {scenset[i]: colors[i] for i in range(len(scenset))}

results = results.sort_values(2100, ascending=False)

# Process files for both before and after 2100
years = np.arange(2010, 2100)
to_plot = results.loc[(results["quantile"]==0.5), :]
labels = to_plot["scenario"]
plt.clf()
if single_plot:
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
legendstr = []
for (j, scen) in to_plot.iterrows():
    printleg = scen["scenario"]
    legendstr.append(printleg)
    plt.plot(
        years, scen.loc[years], c=cdict[scen["scenario"]],
        alpha=0.55, label=printleg
    )

plt.xlabel("Year")
plt.ylabel("Temperature ($^o$C)")

plt.legend(bbox_to_anchor=(1.02, 1))
plt.savefig(plotdir + savestring + "plot0.5quant.png", bbox_inches="tight")
plt.clf()

to_plot["max"] = to_plot[years].max(axis=1)

to_plot.to_csv(summarydir + savestring + summaryname, index=False)

