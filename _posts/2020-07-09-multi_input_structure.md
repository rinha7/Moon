---
layout: post
title:  "여러 개의 Input을 넣는 방법"
date:   2020-07-09
excerpt: "Tensorflow Keras 세번째"
tag:
- Tensorflow 
- Keras
- Computervision
- python
comments: true
---

## Tensorflow v2.0 에서 Keras 모델 구현하기

### 여러 개의 Input을 받는 구조의 모델 구현하기

Keras 모델을 만들면서, 여러개의 입력 또는 여러개의 출력을 넣을 필요가 있는 모델을 만들 때가 있습니다.

함수형 API를 이용하여 입력과 출력이 여러개인 모델을 만들 수 있습니다.


![multi_input](/assets/img/multi_input_structure.png)

위 그림은 입력층이 여러개인 네트워크를 나타냅니다.

Input A는 hidden layer를 거치지 않고 바로 concat layer로 연결됩니다.

Input B는 2개의 hidden layer를 거쳐 concat layer로 연결됩니다. 

위와 같은 네트워크를 구성하기 위한 코드는 다음과 같습니다.

```ㄴ
inputA = keras.layers.Input(shape=[5])
inputB = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(inputB)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([inputA,hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[inputA, inputB], outputs=[output])
```

모델이 복잡해지는 경우에는, name 속성을 이용하여 layer에 이름을 붙이는 것도 좋습니다.

다른 부분들은 이전 내용인 함수형 API와 똑같이 쓰면 되지만, fit 과 evaluate의 경우에는 input을 2개 넣어주어야 합니다. 따라서 compile 및 fit 코드는 다음과 같아집니다

```
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

X_train_A, X_train_B = X_train[:,:5], X_train[:,2:]
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:,:5], X_test[:,2:]
X_new_A, X_new_B = X_test_A[:3],X_test_B[:3]

history = model.fit((X_train_A,X_train_B), y_train, epochs=20, validation_data=((X_valid_A,X_valid_B),y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A,X_new_B))
```

실행 가능한 전체 모델 코드는 다음과 같습니다.

```

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 캘리포니아의 주택 가격 데이터를 sklearn을 통해서 불러옵니다.
housing =fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

### 캘리포니아 주택 가격 데이터 불러오는 코드 ###


'''
Model 작성 코
'''
inputA = keras.layers.Input(shape=[5])
inputB = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30, activation="relu")(inputB)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([inputA,hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.Model(inputs=[inputA, inputB], outputs=[output])



model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

X_train_A, X_train_B = X_train[:,:5], X_train[:,2:]
X_valid_A, X_valid_B = X_valid[:,:5], X_valid[:,2:]
X_test_A, X_test_B = X_test[:,:5], X_test[:,2:]
X_new_A, X_new_B = X_test_A[:3],X_test_B[:3]

history = model.fit((X_train_A,X_train_B), y_train, epochs=20, validation_data=((X_valid_A,X_valid_B),y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A,X_new_B))
```
