
def get_nb_cluster():

    from database.load import ProjectLoader
    from pandas.plotting import table
    import matplotlib.pyplot as plt
    from util import save

    # Load database
    db = ProjectLoader().load_db()
    # # SQL statement
    # Only the neurons that could be used in the analysis (bursting during undir & number of motifs >= 10)
    df = db.to_dataframe("SELECT * FROM cluster WHERE analysisOK=TRUE")
    df.set_index('id')

    df_nb_cluster = df.groupby(['birdID', 'taskName']).count()['id'].reset_index()
    df_nb_cluster = df_nb_cluster.pivot_table('id', ['birdID'], 'taskName')
    df_nb_cluster = df_nb_cluster.fillna(0).astype(int)
    df_nb_cluster.loc['Total'] = df_nb_cluster.sum(numeric_only=True)

    fig, ax = plt.subplots(1, 1)
    plt.title("# of clusters")
    table(ax, df_nb_cluster, loc="center", colWidths=[0.2, 0.2, 0.2]);
    plt.axis('off')
    plt.show()


def plot_cluster_pie_chart(axis, colors, category_column_name):
    pass


category_column_name = 'unitCategoryUndir'

from database.load import ProjectLoader
import matplotlib.pyplot as plt

# Load database
db = ProjectLoader().load_db()
# # SQL statement
df = db.to_dataframe("SELECT * FROM cluster WHERE ephysOK=TRUE")
df.set_index('id')
unit_category = df[category_column_name].dropna()

# Plot pie chart
explode = (0.1, 0)
colors = ['#66b3ff', '#ff9999']
values = [sum(unit_category == 'Bursting'), sum(unit_category == 'NonBursting')]
fig, ax = plt.subplots()
ax.pie(values, explode=explode, colors=colors,
        shadow=True, labels=unit_category.unique(), startangle=90,
        autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))

plt.title('Cluster Category')
ax.axis('equal')

plt.show()

# summary = summary[summary['UnitCategory_Undir'] == 'Bursting']
# labels=UnitCategory.unique()
# print(labels)


# get_nb_cluster()