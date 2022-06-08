import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from konlpy.tag import Okt, Kkma

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('../datasets/medical_qs_numbering.csv')
# print(df.head())
# print(df.category.value_counts())
# df.info()
df = df.dropna(axis= 0)
print(df.head())
print(df.department.value_counts())

df.info()
X = df.medical_qs
Y = df.department

encoder = LabelEncoder()
labeled_Y = encoder.fit_transform(Y) # encoder에 labeled_Y의 data가 저장된다., Y 데이터에 라벨링
# print(encoder.classes_)
# print(labeled_Y[:5])
with open('../output/medical_encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_Y)
okt = Okt() # 형태소 기준으로 나눠주는 함수, 종류가 5개 -> 결국 다 사용해서 5번 돌린다. open korea token

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True) #stem=True-> 원형으로 만들어주기(ex.접었다->접다)
print(X[0])

stopwords = pd.read_csv('../datasets/stopwords.csv', index_col=0) # 아주 기본적인 불용어 리스트 불러오기, 데이터의 종류에 따라 불용어는 달라진다.
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):
        if len(X[j][i]) > 1: # 글자의 크기가 1 이하는 제외
            if X[j][i] not in list(stopwords['stopword']): # 불용어에 속하면 제외
                words.append(X[j][i])
    X[j] = ' '.join(words)
# print(X[1])
token = Tokenizer()
token.fit_on_texts(X) #X 안의 모든 형태소를 찾아서 unique한 값(숫자)을 부여 dict 형태, token 안에 변환된 dict 정보가 저장되어 있다.
tokened_X = token.texts_to_sequences(X) # dict 형태를 번호로 바꿔서 순서대로 list 화
# print(tokened_X[1])
with open('../output/medical_token.pickle', 'wb') as f:
    pickle.dump(token, f)

# 각각의 list의 길이가 다 다르다->문장의 길이를 맞춰야 한다.
wordsize = len(token.word_index) + 1 # token은 1 부터 시작, 0은 우리가 나중에 불필요한 단어 list길이 맞추기 위해 사용
# print(wordsize)
# print(token.word_index)

max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
# print(max)

X_pad = pad_sequences(tokened_X, max) # max 값보다 작은 list는 압에 0을 붙여서 반환
# print(X_pad[:10])

X_train, X_test, Y_train, Y_test = train_test_split(
    X_pad, onehot_Y, test_size=0.1
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save(f'../output/medical_data_max_{max}_word_size{wordsize}', xy)