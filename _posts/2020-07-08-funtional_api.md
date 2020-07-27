---
layout: post
title:  "Functional API를 이용한 복잡한 모델 생성"
date:   2020-07-08
excerpt: "Tensorflow Keras 두번째"
tag:
- Tensorflow 
- Keras
- Computervision
- python
comments: true
---

## Tensorflow v2.0 에서 Keras 모델 구현하기

### 함수형 API를 사용하여 구현하기

1일차에 정리한 내용과는 다르게, 순차적이지 않은 신경망도 있습니다.

대표적으로는, wide & deep 신경망으로 불리는 구조가 있습니다.

![wide_and_deep](/assets/img/wide_and_deep_structure.png)

wide & deep 신경망은 위와 같은 형태로 되어 있으며, 신경망을 깊게 쌓은 Deep 구조와, 신경망을 짧게 쌓은 Wide 구조를 모두 활용가능하다는 장점이 있습니다. 

wide & deep 신경망을 구현한 코드는 다음과 같습니다.

```
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```
코드는 Input 객체를 만들어 shape 과 dtype을 비롯한 모델의 입력에 대한 설명을 저장합니다.

hidden layer 1은 30개의 뉴런과 relu 활성화 함수를 갖습니다. 이 층은 생성되자마자 input과 함께 호출됩니다.

함수형 API는 이 처럼 keras에 층이 연결될 방법을 알려줍니다. hidden layer 2 역시 첫번째 은닉층과 마찬가지로 구성해줍니다.

이후 concatenate 층을 만들고, 함수처럼 호출하여 두 번째 은닉층과 출력이 될 output을 연결합니다.

마지막으로 사용할 입력층과 출력층을 지정한 model을 만듦으로서 함수형 API를 완성합니다.

전체 코드는 다음과 같습니다.

```
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# 캘리포니아의 주택 가격 데이터를 sklearn을 통해서 불러옵니다.
housing =fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

### 캘리포니아 주택 가격 데이터 불러오는 코드 ###

# 복잡한 신경망을 작성합니다.
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

print(y_pred)
```
