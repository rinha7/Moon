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

&nbsp;&nbsp;&nbsp;&nbsp; Plain method와는 다르게, 이 방법들은 조도를 고려하는 방법들이다. 이러한 방법들은 Retinex theory에 기반하여, image를 reflectance와 illumination이라는  2개의 요소로 나누어 처리한다. 하지만 이러한 방법들 역시 결과 이미지가 자연스럽지 않다거나,
지나치게 대조되는 부분이 생기는 등의 문제가 발생합니다. 

#### Deep Learning-based Methods

&nbsp;&nbsp;&nbsp;&nbsp; 딥러닝 기법의 발견 이후, 많은 딥러닝 method들이 제안되었는데, 이 방법들은 대체로 denoising, super-resolution, dehazing등의 기법이었습니다. 이 논문에서 제안하는 low-light enhancement를 딥러닝 방법으로 처음 제안한 것은 LLNet이고,
이후에도 많은 deep learning 방법들이 제안되었습니다. 기존의 방법들은 대체로 noise가 다양한 빛을 통해 각 지역에 다른 영향을 준다는 것을 고려하지 않고 만들어진 "ground-truth" 이미지를 target 삼아 학습하기 때문에, noise나 색의 왜곡이 발생하기도 합니다.

#### Image Denoising Methods

&nbsp;&nbsp;&nbsp;&nbsp; 영상처리, 컴퓨터 비전, 멀티미디어 분야에서 오래 연구되어 온 주제 중 하나인 denosing은 과거에는 이미지의 일반화를 통해 해결하는 경우가 많았습니다. 가장 유명한 방법 중 하나인 BM3D와 WNNM은 테스트 성능은 잘 나오지만, 실제 데이터를 통해 denoising을 수행하면, 결과가 만족스럽지 않게 나옵니다.
최근에는 딥러닝 기반의 method들도 나오고 있는데, autoencoder를 사용하는 방법이나, feed-foward convolution network를 이용하는 방법들도 있습니다. 하지만, 이러한 방법들도 여전히 blind image denoising의 수행에는 어려움을 갖고 있고, 많은 파라미터를 요구하기 때문에 학습에 어려움을 겪습니다. Recurrent를 통해 이러한 문제는 어느정도 해결할 수 있지만,
이미지의 다른 영역이 다른 noise 수준을 갖는다는 것을 명시하고, 이에 대해 처리하는 네트워크는 적습니다.

![kind_fig2](/assets/img/kind/fig2.png)

### Methodology

&nbsp;&nbsp;&nbsp;&nbsp; KinD 네트워크의 목표는 이미지 속에 숨겨진 degradation들을 효과적으로 제거하고, 유연하게 노출과 광량을 조절할 수 있어야 합니다. KinD에서는 이를 위해 네트워크를 reflectance와 illumination 요소를 다루는 2가지 구조로 나누고, 또한 이를 레이어 분리, reflectance 복원, 그리고 illumination 조정의 3가지 파트로 나눌 수 있습니다.
이번 섹션에서는 이러한 네트워크들의 구성과 역할에 대해 다룹니다.

#### Consideration & Motivation

&nbsp;&nbsp;&nbsp;&nbsp; 이 파트에서는 네트워크의 각 구조의 필요성과 개발 동기에 대해 다룹니다.

 1. Layer Decomposition
    Plain method들에 대해 설명하며 문제로 꼽았던 점 중 하나가 실제 조명에 대해 고려하지 않는다는 점이었습니다. 이를 위한 해결책은 조명 정보(illumination information)에서 얻을 수 있습니다. Input으로부터 조명 정보를 잘 뽑아낸다면, degradation을 지울 수 있거나, 복원할 수 있는 detail에 대해 알 수 있습니다.
    Retinex theory에서는 이미지 $I$는 2가지 구성 요소의 집합이라고 할 수 있는데, reflectance $R$과 illumination $L$이 그 구성요소입니다. 
 2. f 


### Experiments
