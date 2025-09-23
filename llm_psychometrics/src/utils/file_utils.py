import pandas as pd
import os

def append_csvs(folder: str, file1: str, file2: str) -> str:
    path1, path2 = os.path.join(folder, file1), os.path.join(folder, file2)
    df1, df2 = pd.read_csv(path1), pd.read_csv(path2)
    combined = pd.concat([df1, df2], ignore_index=True)
    output_path = os.path.join(folder, f"appended_{file1[:-4]}_{file2[:-4]}.csv")
    combined.to_csv(output_path, index=False)
    return output_path