#!/usr/bin/env python3
#
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages

fig = plt.figure()
ax = plt.axes(projection="3d")

# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = 15 * np.random.random(100)
ydata = 15 * np.random.random(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap="viridis")

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

# Save figures
pdf_filename = "../reports/random_3d.pdf"
with PdfPages(pdf_filename) as pdf:
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
# Close all figures
plt.close("all")
