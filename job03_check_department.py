import pandas as pd

df = pd.read_csv('./datasets/medical_qs_all.csv')
print(df['department'].value_counts())