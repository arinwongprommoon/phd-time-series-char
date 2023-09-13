#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

data_dir = "../data/raw/"
group1_name = "is20016_zwf1egf"
group2_name = "is20016_by4741"

filepath1 = data_dir + group1_name
timeseries1_filepath = filepath1 + "_timeseries.csv"
labels1_filepath = filepath1 + "_labels.csv"

timeseries1_df = pd.read_csv(timeseries1_filepath, index_col=[0, 1, 2])

filepath2 = data_dir + group2_name
timeseries2_filepath = filepath2 + "_timeseries.csv"

timeseries2_df = pd.read_csv(timeseries2_filepath, index_col=[0, 1, 2])

timeseries_df = pd.concat([timeseries1_df, timeseries2_df])
timeseries_dropna = timeseries_df.dropna()

nrows = 3
fig, ax = plt.subplots(
    nrows=nrows,
    sharex=True,
    figsize=(6, 6),
)

ax[0].plot(timeseries_dropna.iloc[425])
ax[0].set_title(f"Sample BY4741 time series (1)")

ax[1].plot(timeseries_dropna.iloc[0])
ax[1].set_title(f"Sample zwf1Δ time series (1)")

ax[2].plot(timeseries_dropna.iloc[14])
ax[2].set_title(f"Sample zwf1Δ time series (2)")

for row_idx in range(nrows):
    ax[row_idx].xaxis.set_major_locator(ticker.MultipleLocator(20))

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Time (min)")
plt.ylabel("Flavin fluorescence, normalised (AU)")

# Save figures
pdf_filename = "../reports/sample_ts.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
