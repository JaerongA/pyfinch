"""
Collect SpkInfo.mat files for spike info analysis
"""

def main():

    from ..database.load import ProjectLoader, DBInfo
    import shutil
    from ..utils import save

    # Load database
    db = ProjectLoader().load_db()
    # SQL statement
    query = "SELECT * FROM cluster WHERE analysisOK = 1"
    db.execute(query)

    # Loop through db
    for row in db.cur.fetchall():

        # Load cluster info from db
        cluster_db = DBInfo(row)
        name, path = cluster_db.load_cluster_db()
        mat_file = path / 'SpkInfo.mat'
        new_mat_name = name + '.mat'

        save_path = save.make_dir(ProjectLoader().path / 'Analysis', 'InformationAnalysis', add_date=False)
        new_file_path = save_path / new_mat_name
        shutil.copy(mat_file, new_file_path)


if __name__ == '__main__':
    main()