---
layout: post
title:  "Kindling the Darkness: A Practical Low-light Image Enhancer"
date:   2021-04-22
excerpt: "Kindling the Darkness: A Practical Low-light Image Enhancer 이해하기"
tag:
- python
- tensorflow
comments: true
---

## Kindling the Darkness : A Practical Low-light Image Enhancer

&nbsp;&nbsp;&nbsp;&nbsp; Kindling the Darkness는 Low light image enhancement 기법을 소개하는 논문입니다. 논문 파일의 주소 및 github 주소는 아래와 같습니다.

```  
논문 주소 : https://arxiv.org/pdf/1905.04161.pdf
코드 깃허브 : https://github.com/zhangyhuaee/KinD
```  

### Introduction
&nbsp;&nbsp;&nbsp;&nbsp; high-quality의 이미지를 빛이 부족한 상태에서 찍는 것은 굉장히 어려운 일입니다. 이러한 문제를 해결하기 위해 ISO를 증가시키거나, 노출을 증가시키는 방법이 있지만, 이러한 방법들은 image에 noise가 생기게 합니다. 따라서, low light image enhancement는 굉장히 중요한 분야입니다.
지난 수 년간 low-light image enhancer를 만들기 위한 다양한 시도들이 존재했지만, 밝기에 유연하게 대응하고, degradation을 효과적으로 제거하며, 효율적인 부분까지 모두 고려해야 하므로, 실용적인 low-light enhancer를 개발하는 것은 여전히 어려운 일입니다.

![kind_fig1](/assets/img/kind/fig1.png)
<figcaption style="text-align:center">fig1. kind의 enahancement 결과 예시 </figcaption>

&nbsp;&nbsp;&nbsp;&nbsp; Deep learning 기반의 enhancement 기법들(SR, Denoising 등의)은 기존의 기법들에 비해 훨씬 좋은 성능을 보여주지만, 대부분은 ground truth data를 포함하는 dataset을 필요로 합니다. 실제 low-light image의 경우, 사람들이 선호하는 조도는 각자 다를 수 있기 때문에, 특정 조도에 대해서만 학습하는 것은 유용하지 않습니다.
기존 daataset을 통해 학습하는 방법들에는 이러한 문제가 있었기 때문에, KinD에서는 well-defined ground-truth 없이 2장/혹은 몇몇의 다른 샘플들을 통해서 학습하는 것을 목표로 합니다.

### Previous Method

#### Plain Method

&nbsp;&nbsp;&nbsp;&nbsp; Plain Method란, Histogram Equalization, Gamma Correction(감마 보정) 등의 method들을 말한다. 이런 문제들은 실제 조명 요인을 고려하지 않고, 일반적으로 향상된 결과를 제공하므로, 실제 장면과 일관되지 않는 이미지가 나온다.

#### Traditional Illumination-based Methods

&nbsp;&nbsp;&nbsp;&nbsp; Plain method와는 다르게, 이 방법들은 조도를 고려하는 방법들이다.


### Methodology


### Experiments
