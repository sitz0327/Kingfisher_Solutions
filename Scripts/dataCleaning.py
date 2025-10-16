import os
import pandas as pd
import glob

def FixDateTime(input_path):
    files = glob.glob(os.path.join(input_path, "*.csv"))
    print(files[0])
    df = pd.read_csv(files[0])

    df[['Year','Month','Day','Hour']] = df[['Year','Month','Day','Hour']].astype(int)
    df['datetime'] = pd.to_datetime(df[['Year', 'Month','Day', 'Hour']]) 
    df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'],inplace=True) 
    df.to_csv(files[0], index=False)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(script_dir)
    parent_dir = os.path.join(parent, "Data", "NSRDB")
    print(parent_dir)


    for dir in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, dir) 
        output_path = os.path.join(dir_path, f"{dir}_combined.csv")
        FixDateTime(dir_path)