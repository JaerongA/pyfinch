# from database.load import database
#
# conn, cur = database()


from summary import load


summary_df, nb_cluster = load.summary(load.config())
