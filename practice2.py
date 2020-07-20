from matplotlib.lines import Line2D # Imported for legends
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np

intervals = [(1, 2), (1.1, 2.5), (1.2, 4), (1.5, 10), (1.7, 12)]
num_intervals = len(intervals)
viridis = plt.cm.get_cmap('viridis', num_intervals)
colors = np.array([viridis(idx / num_intervals) for idx in range(len(intervals))])

# Prepare the input data in correct format for LineCollection
lines = [[(i[0], j), (i[1], j)] for i, j in zip(intervals, range(len(intervals)))]

lc = mc.LineCollection(lines, colors= colors, linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.margins(0.1)
plt.yticks([], [])

# Adding the legends
def make_proxy(col, scalar_mappable, **kwargs):
    color = col
    return Line2D([0, 1], [0, 1], color=color, **kwargs)
proxies = [make_proxy(c, lc, linewidth=2) for c in colors]
ax.legend(proxies, range(5))

# Adding annotations
for i, x in enumerate(intervals):
    plt.text(x[0], i+0.1, x[0], color=colors[i])
    plt.text(x[1], i+0.1, x[1], color=colors[i])

plt.show()