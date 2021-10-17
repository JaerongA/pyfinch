"""Plot cluster information (e.g., number of cluster for each bird, categories, etc"""

from database.load import ProjectLoader
import matplotlib.pyplot as plt
from deafening.results.plot import get_nb_cluster
from util import save

# Parameters
save_fig = False
fig_ext = '.png'

fig, ax = plt.subplots(1, 1)
plt.suptitle("# of clusters", y=.9, fontsize=20)
get_nb_cluster(ax)
plt.axis('off')
fig.tight_layout()

# Save results
if save_fig:
    save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
    save.save_fig(fig, save_path, 'ClusterInfo', fig_ext=fig_ext)
else:
    plt.show()


