import pandas as pd
import glob
data_paths = glob.glob('../notwell/*') # notwell 안의 경로를 리스트로
print(data_paths)
for path in data_paths:
    df_temp = pd.read_csv(path) # 각 경로 안의 각각의 csv 파일 읽기
    # print(df_temp)
    l = df_temp.shape[1] #columns 의 개수
    j = int(l/3)
    df = pd.DataFrame()
    for i in range(1,j):  # 1~10
            a = (i - 1) * 3
            b = a+1
            c = b+1
            globals()['family_{}'.format(i)] = df_temp[[str(a), str(b), str(c)]]
            k = globals()['family_{}'.format(i)]
            k.columns = ['title', 'content', 'department']
            # print(globals()['family_{}'.format(i)].head())
            df = pd.concat([df, k], ignore_index=True)
    print(df.head())
    df.info()
    name = path[10:]
    # df.to_csv(f'./notwell/1{name}', index = False)