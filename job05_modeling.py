import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './output/medical_data_max_817_word_size32196.npy',
    allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(32196, 300, input_length=817)) # 12537 =>단어의 개수, 전체의 단어가 12537, 각각의 단어에 값을 부여 12537개의 차원
#차원이 너무 많아지면 차원의 저주=> 차원이 늘어나면 data 간의 간격이 멀어진다 so 차원이 늘어날수록 data 가 많아야 한다. 그래서 300의 차원으로 조절하라
#일반적으로 200~ 300 개 정도의 차원
model.add(Conv1D(32, kernel_size=5, padding='same', activation = 'relu')) # 앞 뒤 간의 위치 관계(순서가 있는 data도 위치 관계를 볼 수 있다.)
# 위치 관계만 볼 경우 빠르게 인식 가능, 분류도 잘 되고 빠르다
model.add(MaxPooling1D(pool_size=1)) # 1개 중 가장 큰 값, 빼도 상관 없다.
model.add(LSTM(256, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(39, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

fit_hist= model.fit(X_train, Y_train, batch_size=100, epochs= 30, verbose=1, validation_data = (X_test, Y_test))

model.save('./output/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))

plt.plot(fit_hist.history['accuracy'], label = 'accuracy')
plt.plot(fit_hist.history['val_accuracy'], label = 'val_accuracy')
plt.legend()
plt.show()