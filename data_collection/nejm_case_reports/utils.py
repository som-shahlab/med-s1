import os
import pandas as pd
from typing import List

def load_collected_urls_df(path_to_input_dir: str) -> pd.DataFrame:
    dfs: List[str] = []
    for file in sorted(os.listdir(path_to_input_dir)):
        # parse "page_<number>.csv" => number
        page_number: int = int(file.split(".")[0].split("_")[1])
        if page_number > 100:
            continue
        if file.endswith(".csv"):
            try:
                dfs.append(pd.read_csv(os.path.join(path_to_input_dir, file)))
            except Exception as e:
                print(f"Error loading {file}: {e}")
    df = pd.concat(dfs)
    return df