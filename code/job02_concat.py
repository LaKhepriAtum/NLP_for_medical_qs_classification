import pandas as pd
import glob
data_paths = glob.glob('../datasets/*') # 데이터set 안의 모든 경로를 리스트로
df = pd.DataFrame()
for paths in data_paths:
    paths = glob.glob(paths) # 데이터 set 안의 path
    for path in paths:
        path = glob.glob(path+'/*')
        for i in path:
            df_temp = pd.read_csv(i)
            # df_temp = df_temp.reset_index()
            df_temp.columns = ['title', 'content', 'department']
            df = pd.concat([df, df_temp], axis='rows')


print(df.head())
df.info()
# #
df.to_csv('./datasets/medical_qs_Classification.csv', index = False)