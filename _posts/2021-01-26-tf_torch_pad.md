---
layout: post
title:  "tensorflow.pad vs torch.nn.funcntional.pad"
date:   2020-07-09
excerpt: "tf.pad와 torch.pad의 차이"
tag:
- Tensorflow 
- Keras
- python
comments: true
---

## Tensorflow와 Torch의 pad 함수 차이

### tf.pad 

torch로 된 코드를 keras로 옮기거나, 한쪽만 보다가 다른 쪽의 코드를 볼 때 헷갈리는 부분을 정리했습니다.

먼저 tensorflow의 pad 함수 설명은 다음과 같습니다.

<a href="https://www.tensorflow.org/api_docs/python/tf/pad"> tf.pad 공식 문서 </a>

![tf_pad](/assets/img/200126/tf_pad.png)

tf.pad의 구성은 다음과 같습니다.

```
tf.pad(
tensor, paddings, mode='CONSTANT', constant_values=0, name=None
)
```

tensor <- 패딩을 수행할 Tensor <br>
paddings <- padding을 수행할 모양 <br>
mode <- padding 방법(CONSTANT, REFLECT, SYMMETRIC) <br>
constant value <- CONSTANT mode로 패딩을 수행할 때 채울 값을 입력할 수 있습니다.
<br>

공식 문서에 나와있는 예시는 다음과 같습니다.
```
t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 1,], [2, 2]])
# 'constant_values' is 0.
# rank of 't' is 2.
tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                 #  [0, 0, 1, 2, 3, 0, 0],
                                 #  [0, 0, 4, 5, 6, 0, 0],
                                 #  [0, 0, 0, 0, 0, 0, 0]]
```

tf.pad의 경우에는 padding에 입력되는 tensor값, 즉 패딩 모양이 입력되는 tensor의 dimension 과 같은 크기여야 합니다.

각 dimension이 padding에 입력되는 것과 같은 모양으로 padding이 수행됩니다. 예시로 주어진 코드의 t는 tensorshape[2,3]의 tensor이고,
paddings은 [[1,1],[2,2]] 입니다. t에 대해 paddings로 tf.pad를 수행하면 [4,7] shape의 tensor가 생성됩니다. 첫번째 dimension에
앞 부분에 1만큼, 뒷 부분에 1만큼 padding이 수행되었고, 2번째 dimension에 앞 부분에 2만큼, 뒷 부분에 2만큼 패딩이 된 것을 확인할 수 있습니다.

이처럼 tf.pad는 paddings가 padding되는 tensor의 dimension과 같아야하고, 앞에서부터 순서대로 padding을 수행하는 것을 알 수 있습니다.

### torch.nn.functional.pad

torch의 pad 역시 같은 역할을 수행하지만 tf pad와는 반대되는 순서로 padding을 수행합니다. 

torch의 pad 함수의 공식 문서는 다음과 같습니다.

<a href="https://pytorch.org/docs/stable/nn.functional.html"> torch.nn.functional 공식 문서 </a>

![torch_pad](/assets/img/200126/torch_pad.png)

```
torch.nn.functional.pad(input, pad, mode='constant', value=0)
```

torch pad의 경우, tf pad와 다르게 dimension 크기를 맞춰 줄 필요는 없습니다. torch.nn.functional.pad의 경우 tf.pad와 마찬가지로 앞부분 padding
크기와 뒷부분 padding 크기를 순서대로 입력받으며, 입력받은 tuple의 첫번째와 두번째 값이 가장 마지막 dimension의 padding의 크기를 결정합니다. 만약 마지막
부분만 padding하고 싶다면 1개의 tuple으로 pad를 지정해도 되며, 가장 마지막 dimension이 아닌 다른 dimension만 padding하고 싶다면 첫 번째와 두번째 값을 0으로 설정하면 됩니다.

예를 들어, tf.pad에 사용했던 예시를 torch에서 수행하면 다음과 같은 모습이 됩니다.

```
t = torch.tensor([[1,2,3],[4,5,6]])
paddings = (2,2,1,1)
torch.nn.functional.pad(t,paddings)
tensor([[0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 0, 0],
        [0, 0, 4, 5, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]])
```

torch의 pad의 경우 pad에 입력되는 값이 tensor가 아니라 tuple 형태여야 합니다.

torch.nn.functional.pad의 공식문서에 나온 코드를 보면 torch의 pad가 어떻게 작동하는지 좀 더 자세하게 알 수 있습니다.

```
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p1d = (1, 1) # pad last dim by 1 on each side
>>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
>>> print(out.size())
torch.Size([3, 3, 4, 4])
>>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
>>> out = F.pad(t4d, p2d, "constant", 0)
>>> print(out.size())
torch.Size([3, 3, 8, 4])
>>> t4d = torch.empty(3, 3, 4, 2)
>>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
>>> out = F.pad(t4d, p3d, "constant", 0)
>>> print(out.size())
torch.Size([3, 9, 7, 3])
```
