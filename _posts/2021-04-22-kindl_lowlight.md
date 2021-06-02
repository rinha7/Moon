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


### Methodology

&nbsp;&nbsp;&nbsp;&nbsp; KinD 네트워크의 목표는 이미지 속에 숨겨진 degradation들을 효과적으로 제거하고, 유연하게 노출과 광량을 조절할 수 있어야 합니다. KinD에서는 이를 위해 네트워크를 reflectance와 illumination 요소를 다루는 2가지 구조로 나누고, 또한 이를 레이어 분리, reflectance 복원, 그리고 illumination 조정의 3가지 파트로 나눌 수 있습니다.
이번 섹션에서는 이러한 네트워크들의 구성과 역할에 대해 다룹니다.

#### Consideration & Motivation

&nbsp;&nbsp;&nbsp;&nbsp; 이 파트에서는 네트워크의 각 구조의 필요성과 개발 동기에 대해 다룹니다.

 1. Layer Decomposition
    Plain method들에 대해 설명하며 문제로 꼽았던 점 중 하나가 실제 조명에 대해 고려하지 않는다는 점이었습니다. 이를 위한 해결책은 조명 정보(illumination information)에서 얻을 수 있습니다. Input으로부터 조명 정보를 잘 뽑아낸다면, degradation을 지울 수 있거나, 복원할 수 있는 detail에 대해 알 수 있습니다.
    Retinex theory에서는 이미지 $I$는 2가지 구성 요소의 집합이라고 할 수 있는데, reflectance $R$과 illumination $L$이 그 구성요소입니다. 
 2. Data Usage & Priors

분해된 reflectance에서, 어두운 illumination 의 오염은 밝은곳에 비해 더 무겁습니다. 수학적으로, 저하된(degraded) low-light 영상은 $I=R \circ L+E$(E는 오염된 요소)로 표현할 수 있습니다. 이를 수학적으로 간단하게 풀면 다음 수식을 얻을 수 있는데  $$I=R \circ L+E= \tilde{R} \circ L = (R+ \tilde{E}) \circ L=R \circ L + \tilde{E} \circ L,$$ 로 표현할 수 있습니다. 여기서 $\tilde{R}$ 은 오염된 reflectance를 말하며, $\tilde{E}$는 분리된 illumination의 degradation을 의미합니다.  예를들어, White Gaussian noise $E \sim \mathcal{N}(0,\sigma^2)$를 적용한다면 $\tilde{E}$의 분포는 더 복잡해질 것이고, $L$에 더 강하게 영향을 받게 됩니다.    이는 즉, reflectance의 복원은 단순하게 일괄적으로 적용할 수 있는 것이 아니며, illumination map이 reflectance 복원에 좋은 guide 역할을 해줄 수 있다는 것을 의미합니다.(어떻게?)  그렇다면, 직접적으로 $I$에서 $\tilde{E}$ 를 지우는 방법을 사용하는 것에 대해 생각해볼 수도 있습니다. 한 가지 이유로, 불균등 문제가 여전히 남아있습니다.  다른 관점에서 보자면, 고유한 세부 요소들이 noise들과 혼동될 수 있습니다. reflectance 외의 점으로는, 다양한 L(illumination) degradation 제거에 있어서 적절한 reference를 가지고 있지 않기 때문에, 유사한 분석을 하게 되는데, 이는 color-distortion과 같은 다른 유형의 degradtion을 제공하게 됩니다.
 3. Illuminatoin Guided Reflectance Restoration.
    분해된 reflectance에서, 어두운 illumination 의 오염은 밝은곳에 비해 더 무겁습니다. 수학적으로, 저하된(degraded) low-light 영상은 $I=R \circ L+E$(E는 오염된 요소)로 표현할 수 있습니다. 이를 수학적으로 간단하게 풀면 다음 수식을 얻을 수 있는데
$$$I=R \circ L+E= \tilde{R} \circ L = (R+ \tilde{E}) \circ L=R \circ L + \tilde{E} \circ L,$$$
    로 표현할 수 있습니다. 여기서 $\tilde{R}$ 은 오염된 reflectance를 말하며, $\tilde{E}$는 분리된 illumination의 degradation을 의미합니다.
    예를들어, White Gaussian noise $E ~ \mathcal{N}(0,\sigma^2)$를 적용한다면 $\tilde{E}$의 분포는 더 복잡해질 것이고, $L$에 더 강하게 영향을 받게 됩니다.
    이는 즉, reflectance의 복원은 단순하게 일괄적으로 적용할 수 있는 것이 아니며, illumination map이 reflectance 복원에 좋은 guide 역할을 해줄 수 있다는 것을 의미합니다.(어떻게?)
    그렇다면, 직접적으로 $I$에서 $\tilde{E} 를 지우는 방법을 사용하는 것에 대해 생각해볼 수도 있습니다. 한 가지 이유로, 불균등 문제가 여전히 남아있습니다.
    다른 관점에서 보자면, 고유한 세부 요소들이 noise들과 혼동될 수 있습니다. reflectance 외의 점으로는, 다양한 L(illumination) degradation 제거에 있어서 적절한 reference를 가지고 있지 않기 때문에, 유사한 분석을 하게 되는데, 이는 color-distortion과 같은 다른 유형의 degradtion을 제공하게 됩니다.
 4. Arbitrary Illumination Manipulation
    사람마다 선호하는 조명 강도(illumination strength)는 다양할 수 있습니다. 그러므로, 실제 시스템은 임의의 조명 조작을 위한 인터페이스를 제공할 필요가 있습니다.
    논문에서는 조명 세기를 향상시키기 위해서 3가지 방법을 주로 사용하는데 fusion, light level appointment, gamma correction 이 이에 해당됩니다.
    Fusion 기반의 방법들은 fixed fusion mode로 인해서 빛의 세기를 조정하는 기능이 부족합니다.(왜??) 두 번째 옵션을 채택할 경우(light level appointment를 말하는듯) 데이터 셋에는 목표 수준의 영상들이 포함되어야 하므로
    학습의 유연성이 제한되게 됩니다. Gamma correction의 경우, 이는 각각 다른 $\gamma$값들을 통해서 목표에 도달할 수 잇찌만, light level과의 상관관계를 반영하는 것은 불가능합니다.
    이 논문은 사용자가 임의의 빛과 노출 수준을 지정할 수 있도록 허용하는 유연한 mapping function을 실제 데이터를 통해 학습할 수 있도록 합니다.


![kind_fig2](/assets/img/kind/fig2.PNG)
    
#### KinD Network

위에 소개한 consideration & motivation을 바탕으로, 논문의 저자들은 KinD라는, kindling the darkness deep neural network를 설계하였습니다. 이 아래에는 3개의 subnet에 대한 설명과 기능적 관점에 대한 세부사항에 대하여 묘사합니다.

![kind_fig4](/assets/img/kind/fig4.PNG)

1. Layer Decomposition Net  
   하나의 이미지로부터 2개의 구성요소를 복원하는 것은 설정이 잘못된 문제라고 할 수 있습니다. Ground-truth 정보가 없다면, 잘 디자인 된 제한적인 로스가 중요합니다. 따라서, KinD에서는 2개의 다른 빛 세기 / 노출 정도를 가진 이미지 $[I_l,I_h]$를 준비합니다.
   특정 장면의 reflectance는 설 다른 image들에 거쳐 공유해야한다는 것을 생각하면, 분해한 reflectance pair $[R_l,R_h]$는 비슷해야 합니다.  더 나아가서, illumination map $[L_l,L_h]$ 는 각각이 매끄럽고, 서로간에 일관성이 있어야 합니다.
   이를 위해서 논문에서는 Loss ${L_{is}^{LD} := \lVert R_l - R_h\rVert_1 }$ 을 사용하여 반사율 유사도(reflectance similarity)를 정규화합니다.(${\lvert \rvert_1}$ 은 $l_1$ norm을 의미함) 
   
   Illumination의 smoothness는 ${L_{is}^{LD} := {\lVert {\nabla L_I \over max(| \nabla I_l , \epsilon |)}\rVert}_1 + {\lVert {\nabla L_h \over max(| \nabla I_h , \epsilon |)}\rVert}_1}$ $\nabla$는 $\nabla x$ 와 $\nabla y$의 방향을 포함하는 1차 미분 연산을 의미합니다.
   거기에 $\epsilon$은 0.001 정도의 작은 상수값을 넣어 0으로 나누어지는걸 방지하고,  $| \cdot |$는 절대값을 의미합니다. 이 smoothness term은 입력에 대한  illumination 구조의 연관성을 측정합니다.
   만약 $I$의 edge부분이라면, penalty loss $L$은 작은 값이 될 것이고, 평평한 부분이라면, 페널티가 커지게 됩니다.
   
   상호 일관성에 의해서, $\mathcal{L}_{mc}^{LD}:={\lVert M\circ exp(-c\cdot M) \rVert}_1$ 이라는 Loss를 채택했고 여기서 $M$은 $M := \lvert \nabla L_l \rvert+\lvert \nabla L_h \rvert$를 말합니다.
   Figure 4는 $u \cdot exp(-c \cdot u)$ 는 함수의 동작에 대해 보여주고 있으며, 여기서 $c$는 함수의 모양을 제어하는 파라미터입니다. Figure 4를 통해 볼 수 있듯이, penalty가 처음에는 증가하다가 $u$가 증가함에 따라 0을 향해 떨어집니다.
   
   이러한 특성이 강한 상호 모서리?(원문 : mutual edge)가 약한 쪽에 비해 학습에서도 보존되는 일관성(mutual consistency)에 적합하다고 할 수 있습니다. 그리고 c=0으로 설정하는 것은 M을 단순한 $l^1$ loss의 형태로 만드는 것을 알 수 있었습니다.
   게다가, 분리된 2개의 레이어가 reconstruction error($\mathcal{L}^{LD}_{rec}:=  {\lVert {I_l - R_l \circ L_l}  \rVert }_1+{\lVert {I_h - R_h \circ L_h}  \rVert }_1$)에 의해 제한된 입력 이미지를 재구성해야합니다.
   결과적으로 decomposition net의 총 loss는 다음과 같게 됩니다.
   
   $$ \mathcal{L}^{LD}:= \mathcal{L}^{LD}_{rec}+0.01\mathcal{L}^{LD}_{rs}+0.15\mathcal{L}^{LD}_{is}+0.2\mathcal{L}^{LD}_{mc}$$

   decomposition network의 레이어는 각각 reflectance와 illumination에 대응하는 가지들을 포함하고 있으며, reflectance 를 담당하는 부분은 전형적인 5-layer U-Net (conv와  sigmoid) 구조로 되어있고,
   illumination을 담당하는 부분은 2개의 conv+ReLU layer들과 reflectance 가지에서 넘어오는 부분과 합치는 conv와 concat 레이어로 구성(illumination으로부터 texture를 분리하기 위해서(??))되어 있습니다.
   이는 최종적으로 conv+ sigmoid  layer로 합쳐져 illumination을 분해하는 역할을 수행합니다.

![kind_fig3](/assets/img/kind/fig3.PNG)

![kind_fig5](/assets/img/kind/fig5.PNG)

 2. Reflectance Restoration Net
   Low image에서 뽑은 reflectance map들은, Figure 3과 Figure 5에서 보이는 것처럼 밝은 이미지에 비해 degradation의 영향을 더 많이 받습니다. 
    
    
### Experiments
