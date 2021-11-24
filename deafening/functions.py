"""Analysis functions for deafening project"""


def get_pre_post_mean_per_bird(df, variable):
    """
    Get conditional mean for each bird
    Parameters
    ----------
    df : dataframe
    variable : str
        target column name to calculate the mean

    Returns
    -------
    df_mean : dataframe
        new data frame with the computed mean
    """
    import pandas as pd
    df_mean = df.groupby(['birdID', 'taskName']) \
        [[variable]].mean().reset_index()
    df_mean = pd.pivot_table(data=df_mean, index=['birdID'], columns=['taskName'])[variable]
    df_mean = df_mean.reindex(['Predeafening', 'Postdeafening'], axis=1)

    return df_mean
