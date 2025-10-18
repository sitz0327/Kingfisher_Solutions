import os
import pandas as pd
import glob

def concatFiles(file_list, output_path):
    df = pd.read_csv(file_list[0])
    for f in file_list[1:]:
        print(f)
        temp = pd.read_csv(f)
        df = pd.concat([df,temp], ignore_index=True)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    parent_dir = os.path.join(parent, "Data", "MergedSiteData")
    output_path = os.path.join(parent, "Data", "DB.csv",)
    print(parent_dir)
    files = glob.glob(os.path.join(parent_dir, "*.csv"))

    concatFiles(files, output_path)
