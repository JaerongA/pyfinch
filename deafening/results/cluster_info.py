"""Plot cluster information (e.g., number of cluster for each bird, categories, etc"""
from util import save
from database.load import ProjectLoader
import matplotlib.pyplot as plt


def get_nb_clusters(save_fig=False, fig_ext='.png'):
    from deafening.plot import get_nb_cluster

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


def plot_cluster_pie_chart(save_fig=False, fig_ext='.png'):
    # Load database
    db = ProjectLoader().load_db()
    # # SQL statement
    df = db.to_dataframe("SELECT unitCategoryUndir FROM cluster WHERE ephysOK=TRUE")
    unit_category = df['unitCategoryUndir']

    explode = (0.1, 0)
    colors = ['#66b3ff', '#ff9999']
    values = [sum(unit_category == 'Bursting'), sum(unit_category == 'NonBursting')]

    fig, ax = plt.subplots()
    ax.pie(values, explode=explode, colors=colors,
           shadow=True, labels=unit_category.unique(), startangle=90,
           autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))

    plt.title('Unit Category (Undir)')
    ax.axis('equal')

    # Save results
    if save_fig:
        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'Results')
        save.save_fig(fig, save_path, 'ClusterInfo', fig_ext=fig_ext)
    else:
        plt.show()


get_nb_clusters()
plot_cluster_pie_chart()
