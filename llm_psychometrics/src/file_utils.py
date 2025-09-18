import pandas as pd
import os

def join_csvs(folder: str, file1: str, file2: str, join_key: str, how: str = "inner") -> str:
    """
    Reads two CSV files, joins them on a given key,
    and writes the result back to the same folder.

    Args:
        folder (str): Path to the folder containing the CSVs.
        file1 (str): First CSV filename.
        file2 (str): Second CSV filename.
        join_key (str): Column name to join on.
        how (str): Type of join: 'inner', 'left', 'right', or 'outer'. Default is 'inner'.

    Returns:
        str: Path to the merged CSV.
    """
    path1 = os.path.join(folder, file1)
    path2 = os.path.join(folder, file2)

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    merged = pd.merge(df1, df2, on=join_key, how=how)

    output_file = f"merged_{file1[:-4]}_{file2[:-4]}.csv"
    output_path = os.path.join(folder, output_file)

    merged.to_csv(output_path, index=False)
    return output_path