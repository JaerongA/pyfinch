from database.load import ProjectLoader
import matplotlib.pyplot as plt
from util import save
import pandas as pd

# Load database
db = ProjectLoader().load_db()
# SQL statement
# query = "SELECT * FROM pcc"
# db.execute(query)

df = db.to_dataframe(query = "SELECT * FROM pcc")
pcc_undir_sig = df['pccUndirSig']
pcc_dir_sig = df['pccDirSig']
task_list = pd.unique(df.taskName).tolist()
explode = (0.1, 0)
colors = ['#66b3ff', '#ff9999']
values = [55, 6]

fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].pie(values, explode=explode, colors=colors,
        shadow=True,
        labels=['sig', 'non-sig'],
        startangle=90,
        autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))
axes[0].set_title('PCC sig (Undir)')

values = [55, 6]
axes[1].pie(values, explode=explode, colors=colors,
        shadow=True,
        labels=['sig', 'non-sig'],
        startangle=90,
        autopct=lambda p: '{:.2f}%  ({:,.0f})'.format(p, p * sum(values) / 100))
axes[1].set_title('PCC sig (Dir)')

axes[1].axis('equal')
axes[1].axis('equal')

plt.show()