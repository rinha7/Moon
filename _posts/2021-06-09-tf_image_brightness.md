---
layout: post
title:  "tensorflow.image에서 밝기를 조절하는 메소드 3가지"
date:   2021-06-09
excerpt: "tensorflow image 밝기 조절"
tag:
- Tensorflow 
- Torch
- python
comments: true
---

## Tensorflow.image의 밝기 조절 메소드

### tf.image.adjust_brightness

<a href="https://www.tensorflow.org/api_docs/python/tf/image/adjust_brightness">https://www.tensorflow.org/api_docs/python/tf/image/adjust_brightness</a>
```
tf.image.adjust_brightness(
    image, delta
)
```
adjust_brightness는 이미지의 밝기를 조정할 수 있는 tf keras 함수입니다. 이 함수는 RGB 이미지를 float의 형태로 변경하며, 밝기를 조정하고 다시 원래의 data type으로 바꿉니다.
image는 밝기를 변환할 이미지를 넣으면 되고, delta는 image에 더해질 tensor 값을 의미합니다. delta값은 (-1,1) 사이의 값이어야 합니다.

```
x = [[[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0]]]
tf.image.adjust_brightness(x,delta=0.1)

<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[ 1.1,  2.1,  3.1],
        [ 4.1,  5.1,  6.1]],
       [[ 7.1,  8.1,  9.1],
        [10.1, 11.1, 12.1]]], dtype=float32)>
```

위는 tf 페이지에서 제공하는 예시입니다. 이번에는 실제 uint8 RGB 이미지를 불러와서 밝기를 조정해보도록 하겠습니다


```
<tf.Tensor: shape=(3000, 4000, 3), dtype=uint8, numpy=
array([[[ 32,  81, 150],
        [ 34,  83, 152],
        [ 39,  88, 157],
        ...,
        [ 36,  94, 167],
        [ 34,  92, 165],
        [ 33,  91, 164]],
       [[ 32,  81, 150],
        [ 34,  83, 152],
        [ 36,  86, 155],
        ...,
        [ 36,  94, 167],
        [ 36,  95, 165],
        [ 35,  94, 164]],
       [[ 31,  81, 150],
        [ 34,  84, 153],
        [ 34,  84, 153],
        ...,
        [ 37,  94, 165],
        [ 38,  97, 167],
        [ 38,  97, 167]],
       ...,
       [[158, 134, 108],
        [134, 110,  86],
        [124, 100,  76],
        ...,
        [149, 143, 143],
        [154, 150, 149],
        [159, 155, 154]],
       [[132, 106,  81],
        [127, 101,  78],
        [122,  96,  73],
        ...,
        [141, 133, 131],
        [142, 136, 136],
        [145, 139, 139]],
       [[126,  99,  72],
        [128, 100,  76],
        [122,  94,  70],
        ...,
        [138, 130, 128],
        [136, 128, 126],
        [135, 127, 125]]], dtype=uint8)>
```

위는 제가 가지고 있는 image 파일을 tf.io.read_file로 읽어와서 decode를 통해 uint8 이미지로 변환한 tensor입니다. 여기에 adjust_brightness를 적용하면 다음과 같아집니다.
```
tf.image.adjust_brightness(image,delta=0.1)
<tf.Tensor: shape=(3000, 4000, 3), dtype=uint8, numpy=
array([[[ 57, 106, 175],
        [ 59, 108, 177],
        [ 64, 113, 182],
        ...,
        [ 61, 119, 192],
        [ 59, 117, 190],
        [ 58, 116, 189]],
       [[ 57, 106, 175],
        [ 59, 108, 177],
        [ 61, 111, 180],
        ...,
        [ 61, 119, 192],
        [ 61, 120, 190],
        [ 60, 119, 189]],
       [[ 56, 106, 175],
        [ 59, 109, 178],
        [ 59, 109, 178],
        ...,
        [ 62, 119, 190],
        [ 63, 122, 192],
        [ 63, 122, 192]],
       ...,
       [[183, 159, 133],
        [159, 135, 111],
        [149, 125, 101],
        ...,
        [174, 168, 168],
        [179, 175, 174],
        [184, 180, 179]],
       [[157, 131, 106],
        [152, 126, 103],
        [147, 121,  98],
        ...,
        [166, 158, 156],
        [167, 161, 161],
        [170, 164, 164]],
       [[151, 124,  97],
        [153, 125, 101],
        [147, 119,  95],
        ...,
        [163, 155, 153],
        [161, 153, 151],
        [160, 152, 150]]], dtype=uint8)>
```

tensor 내부의 값들이 전체적으로 올라간 것을(밝아진 것을) 확인할 수 있습니다.

tf.image.adjust_brightness는 delta 값을 통해 image의 밝기를 조정합니다. 이외에 augmentation을 위해 random으로 밝기를 조정하고 싶을 때 사용하는 밝기 조정 메소드도 존재합니다.

### tf.image.random_brightness

<a href="https://www.tensorflow.org/api_docs/python/tf/image/random_brightness">https://www.tensorflow.org/api_docs/python/tf/image/random_brightness</a>
```
tf.image.random_brightness(
    image, max_delta, seed=None
)
```


adjust_brightness와 비슷한 역할을 수행하지만, random으로 밝기 값을 조정해줍니다.

```
x = [[[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]],
     [[7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0]]]
tf.image.random_brightness(x,0.2)

<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[ 0.8424369,  1.8424369,  2.8424368],
        [ 3.8424368,  4.842437 ,  5.842437 ]],
       [[ 6.842437 ,  7.842437 ,  8.842437 ],
        [ 9.842437 , 10.842437 , 11.842437 ]]], dtype=float32)>
```

### tf.image.stateless_random_brightness

<a href="https://www.tensorflow.org/api_docs/python/tf/image/stateless_random_brightness">https://www.tensorflow.org/api_docs/python/tf/image/stateless_random_brightness</a>
```
tf.image.stateless_random_brightness(
    image, max_delta, seed
)
```

tf.image.random_brightness는 밝기값을 random으로 조정해주기는 하지만, 모든 시행에 대해 같은 값을 적용합니다. stateless_random_brightness의 경우에는, 모두 동일한 값이 아니라 독립적인 random 값을 부여합니다.

```
x = [[[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0]],
     [[7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0]]]
      
tf.image.stateless_random_brightness(x,0.2,(1,2))
<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
array([[[ 1.1376241,  2.1376243,  3.1376243],
        [ 4.1376243,  5.1376243,  6.1376243]],
       [[ 7.1376243,  8.137624 ,  9.137624 ],
        [10.137624 , 11.137624 , 12.137624 ]]], dtype=float32)>
```
