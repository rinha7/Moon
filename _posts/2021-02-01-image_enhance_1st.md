---
layout: post
title:  "CNN을 이용한 Image Enhancement"
date:   2021-02-01
excerpt: "CNN을 이용한 Image Enhancement"
tag:
- Tensorflow 
- Keras
- python
- IE
comments: true
---

## Image Enhancement란?

![unet_structure](/assets/img/210202/image_enhance_example.png)
<figcaption>Image Enhancement 예시</figcaption>

이미지 Enhancement는 말 그대로 저화질의 이미지를 고화질로 바꾸어주는 것을 말합니다. 해상도를 높여주는 Super Resolution이나, 안개를 제거하는 Denoising,
흔들림을 보정하는 Deblurring등의 다양한 영상 개서 기법들이 여기에 속할 수 있습니다.

제가 다룰 Enhancement 기법들은 예시로 든 방법들과는 다른 방법들입니다. 저화질 이미지를 전문가가 보정한 것과 같은 이미지로 만들어주는 것을 목표로 하며,
데이터셋으로도 실제 전문가들이 사진 5천장을 직접 보정한 Adobe5k dataset을 사용합니다. 해당 데이터셋을 제공하는 사이트는 아래 링크를 통해 접속할 수 있습니다.

<a href="https://data.csail.mit.edu/graphics/fivek/"> MIT-Adobe FiveK Dataset </a>


## 딥러닝을 통한 이미지 향상

![unet_structure](/assets/img/210202/UnetStructure.png)

이미지를 향상시킬 수 있는 가장 기본적인 cnn 구조로 U-Net 구조가 있습니다. U-Net 은 의료영상에서 객체 분할을 위해 제안된 구조인데, 대칭적인 모양의 
Contracting Path와 Expansive Path가 특징인 구조입니다.

U-Net의 이러한 특징을 살려 Enhancement에 사용하는 것도 가능합니다. Adobe5K Dataset을 이용하여 Image Enhancement 학습을 진행하는 네트워크들 중
많은 수의 네트워크들이 U-Net을 기본구조로 사용합니다. 대표적으로 <a href="https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Deep_Photo_Enhancer_CVPR_2018_paper.pdf">Deep Photo Enhancer(2018)</a>나
<a href="https://openaccess.thecvf.com/content_CVPR_2020/papers/Moran_DeepLPF_Deep_Local_Parametric_Filters_for_Image_Enhancement_CVPR_2020_paper.pdf">Deep Local Parametric Filters(2020)</a>등이 있습니다.

기본적인 U-Net구조로 Enhancement를 수행해볼 수 있는 코드는 다음과 같습니다. "data" 디렉토리를 생성하고, 안에 Adobe5K dataset을 넣어 학습을 수행해볼 수 있습니다.

<a href="https://github.com/rinha7/Image_Enhancement_Unet">Image Enhancement Unet Repository</a>

위 코드에 사용된 Network의 구조는 다음과 같습니다.
