import os
import pandas as pd
import glob

def ConcatFiles(input_path, output_path):
    print(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    files = glob.glob(os.path.join(input_path, "*.csv"))

    df = pd.read_csv(files[0], skiprows=2)
    
    i = 0
    for f in files[1:]:
        temp = pd.read_csv(f,skiprows=2)
        df = pd.concat([df,temp], ignore_index=True)
        df.to_csv(os.path.join(output_path), index=False)

    df.to_csv(output_path, index=False)
    return


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    parent_dir = os.path.join(parent, "Data", "NSRDB")
    print(parent_dir)


    for dir in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, dir) 
        output_path = os.path.join(dir_path, f"{dir}_combined.csv")
        ConcatFiles(dir_path, output_path)