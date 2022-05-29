import pandas as pd
import glob
data_paths = glob.glob('./datasets/*')
df = pd.DataFrame()
for paths in data_paths:
    paths = glob.glob(paths)
    for path in paths:
        path = glob.glob(path+'/*')
        for i in path:
            df_temp = pd.read_csv(i)
            df = pd.concat([df, df_temp], ignore_index=True, axis='rows')

# df.columns = ['title', 'content', 'department']
print(df.head())
df.info()

