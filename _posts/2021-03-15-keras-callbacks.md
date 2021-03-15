---
layout: post
title:  "Keras 학습에 사용되는 Callback 4가지"
date:   2021-03-15
excerpt: "keras callbacks api"
tag:
- python
- keras
comments: true
---

## Keras Callbacks

&nbsp;&nbsp;Keras의 callback들은 training 단계에서(epoch 시작부터 끝까지) 어떠한 동작을 수행하는 object들을 말합니다. callback들을 통해서 tensorboard에 모든 batch of training들에 대해 metric 수치를 모니터링할 수도 있고, 이를 저장하는 것도 가능합니다.
Early Stop이나 Learning Rate Scheduling과 같은 기능을 통해 학습결과에 따라 학습을 멈추거나 학습률을 조정할수도 있습니다.
이처럼 Callback들을 잘 활용한다면, 딥러닝 학습의 결과를 보다 좋게 만들 수 있기 때문에, 많이 사용되는 callback 4가지를 소개하고, 사용법에 대해 포스팅하였습니다.

<a href="https://keras.io/api/callbacks/">keras official documentation Callbakcs API</a> 

### LearningRateScheduler

```angular2html
tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```
&nbsp;&nbsp; LearningRateScheduler는 epoch에 따라 학습률을 조정하는 callback입니다. 인자로 받는 schedule은 epoch index를 조정할 수 있는 function을 의미합니다. 
사용 예시는 다음과 같습니다.
```angular2html
def scheduler(epoch, lr):
   if epoch < 10:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=15, callbacks=[callback], verbose=0)
```

&nbsp;&nbsp; 위의 예제에서는 epoch가 10보다 작을 경우에는 입력받은 learning rate를 그대로 사용하고, 10을 넘어가게 되면 lr을 줄여주는 연산을 수행합니다.
 일반적으로 학습이 진행되며 lr을 줄여나가기 때문에, 이런 방법을 사용하는것도 괜찮지만, 학습의 결과를 반영할 수 없고, 학습 시작전에 정한 값을 통해 lr을 줄여야한다는 단점이 있습니다.
다음에 소개할 callback은 이러한 단점을 어느정도 보완하였습니다.

### ReduceLROnPlateau
```
tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
    **kwargs
)
```
&nbsp;&nbsp; Plateau는 안정기를 뜻합니다. ReduceLROnPlateau는 말 그대로 학습이 진행되지 않는 안정기에 들어서면, learning rate에 변화를 줍니다. 동작하는 방법은 간단합니다. 관찰 대상으로 삼은 metric이 정해진 epoch동안 일정 크기 이상 변화하지 않으면 lr을 정해진대로 변경합니다. 파라미터의


monitor는 기준이 될 metrics를 의미하고, <br>factor는 lr에 변화를 어느 정도 줄 것인지, <br>patience는 몇 epoch동안 변화가 없으면 lr을 변화시킬건지,<br>mode는 auto, min, max 3가지가 존재하는데, monitor로 정한 metric이 감소하는 방향으로 움직여야하는지, 증가하는 방향으로 움직여야하는지 정할 수 있습니다.


사용 예시는 다음 코드와 같습니다.

```
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

### ModelCheckPoint
```
tf.keras.callbacks.ModelCheckpoint(
    filepath,
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
    options=None,
    **kwargs
)
```
&nbsp;&nbsp; ModelCheckPoint는 어떤 시점에서 model이나 weights를 저장할 것인지 정할 수 있는 callback입니다. monitor로 넣은 metrics의 변화에 따라, best값을 저장하거나, model을 통째로 저장하거나, weights만을 저장하는 것도 가능합니다.
사용 예시는 다음과 같습니다.

```
model.compile(loss=..., optimizer=...,
              metrics=['accuracy'])

EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

# 지금까지의 학습에서 가장 높은 acc를 보인 모델이 저장됩니다.
model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])


```

### EarlyStopping

```
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
```
&nbsp;&nbsp; EarlyStopping은 metric이 일정 epoch 이상 변화하지 않으면, 학습을 멈추는 callback입니다. min_delta는 변화를 감지할 최소변화량을 의미합니다. min_delta로 지정한 값보다 metric의 변화가 적으면, count가 올라갑니다.
patience는 변화가 어느 정도 없을 때 학습을 멈출 것인지 정하는 파라미터입니다. mode는 metric의 변화 방향을 지정합니다.

EarlyStopping의 사용 예시는 다음과 같습니다.

```
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# 이 callback은 3개의 연속되는 epoch에서 validation loss의 변화가 없을 때  
# 학습을 중단합니다.
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
model.compile(tf.keras.optimizers.SGD(), loss='mse')
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
                    epochs=10, batch_size=1, callbacks=[callback],
                    verbose=0)
```
