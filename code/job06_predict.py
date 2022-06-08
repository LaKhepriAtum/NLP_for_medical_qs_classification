import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model
from konlpy.tag import Okt




pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('../crawling_data/medical_qs_Random.csv')
print(df.head())


X= df.title
Y= df.department

# encoder = LabelEncoder()   이렇게 만들지말고 저장했던 거 불러와야지
with open('../output/encoder_numbering.pickle', 'rb') as f:
    encoder = pickle.load(f)

labeled_Y= encoder.transform(Y)
'''fit_transform은 y를 주면 y의 유니크한 값을 찾음 그럼 카테고리 6개지?  근데 그거 정했잖아.
그거 정한 걸로 그대로 오늘꺼 써야하잖아. 이미 정해진 걸로 라벨링만 작업할거면 transform해야함.'''
'''#라벨을 주고 그 라벨이 뭐냐 물어볼때는 리버스트랜스폼'''

print(encoder.classes_)
print(labeled_Y[:5])





onehot_Y = to_categorical(labeled_Y)
# print(onehot_Y)

okt = Okt()
# x = []
for i in range(len(X)):
   X[i] = okt.morphs(X[i], stem=True)

stopwords= pd.read_csv('../datasets/stopwords.csv', index_col=0)

for j in range(len(X)):
   words =[]
   for i in range(len(X[j])):
      if len(X[j][i]) > 1:  #두글자부터만 보겠다
         if X[j][i] not in list(stopwords['stopword']):  #그중에서도 불용어에 포함 안 되는 애들만
            words.append(X[j][i])
   X[j] = ' '.join(words)
print(X[1])



# 학습 안 한 단어가 오면 0이 옴.

with open('../output/medical_token_numbering.pickle', 'rb') as f:
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)

for i in range(len(tokened_X)):
    if len(tokened_X[i]) > 816: #맥스값
        tokened_X[i] = tokened_X[i][:816]
X_pad = pad_sequences(tokened_X, 816)
label = encoder.classes_
model = load_model('../output/medical_qs_classification_model_0.6597162485122681_numbering.h5')
preds = model.predict(X_pad)
predicts =[]
for pred in preds:
    predicts.append(label[np.argmax(pred)])

df['predict'] = predicts
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'department'] == df.loc[i, 'predict']:
        df.loc[i,'OX'] = 'O'
    else:
        df.loc[i,'OX'] = 'X'
print(df['OX'].value_counts()/len(df))

for i in range(len(df)):
    if df['department'][i] != df['predict'][i]:
        print(df.iloc[i])
