import pandas as pd
df = pd.read_csv('./datasets/medical_qs_all.csv')
df = df.dropna(axis=0)
df.info()
df = df[['medical_qs', 'department']]
df.info()
df.to_csv('./datasets/medical_qs_all.csv', index = False)