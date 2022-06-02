import pandas as pd


well = pd.read_csv('./datasets/concat_all_well.csv')
nunwell = pd.read_csv('./datasets/concat_all_nunwell.csv')

df= pd.concat([well, nunwell], ignore_index=True, axis='rows')
df.info()
df.to_csv('./datasets/final_concat.csv', index=False)