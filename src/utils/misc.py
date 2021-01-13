from pandas.core.frame import DataFrame


def save_df(df: DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
