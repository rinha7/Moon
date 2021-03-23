
---  
layout: post  
title:  "Learning Enriched Features for Real Image Restoration and Enhancement 이해하기"  
date:   2021-03-17  
excerpt: "ECCV 2020 논문 mirnet 이해하기"  
tag:
- Tensorflow
- Keras
- python
- IE  
  comments: true
---  

## Learning Enriched Features for Real Image Restoration and Enhancement

&nbsp;&nbsp;&nbsp;&nbsp; MIRNet이라고 이름 붙인 이 네트워크 구조는 2020년 ECCV에 소개된 논문 Learning Enriched Features for Real Image Restoration and Enhancement 에서 제안한 구조입니다.  
논문의 링크와 제작자의 코드 깃허브 링크는 아래와 같습니다.

```  
논문 주소 :https://arxiv.org/pdf/2003.06792v2.pdf코드 깃허브 : https://github.com/swz30/MIRNet
```  

### Introduction
&nbsp;&nbsp;&nbsp;&nbsp; 이 논문의 저자들은 기존의 Image Enhancement 구조인 Encoder-Decoder 구조와 high-resolution(single-scale) 구조들의 문제점을 지적합니다. 기존 네트워크의 문제점들이란, 제한된 수용영역(receptive fields)에서 맥락과 관련된 정보들(contextual information)을 뽑아내는데 제한적이라는 것입니다. 따라서 이러한 수용 영역을 늘리기 위해, 이 논문에서는 multi-scale approach를 사용합니다. 고해상도에서의 특징맵을 계층적으로 쌓고, 공간적 세부 loss들을 최소화 하는 방향으로 학습합니다. 동시에, parallel convolution streams를 구축하여, multi-resolution feature들이  
교류하며 정보를 교환합니다.

이 논문에서 제시하는 방식이 기존의 방식들과 다른 점은, multi-scale로 얻어낸 맥락 특징들(contextual information)을 종합하는 방식에 있습니다. 기존의 방식들은 대체로 각 scale에서 얻어낸 정보를 독립적으로 취급하거나, top-down 방식으로만 데이터를 종합합니다. 이와는 반대로, 이 MIRNet 모델에서는 top-down, bottom-up 방식의 교류가 모두 이루어지며, 동시에 coarse-to-fine, fine-to-coarse의 교환 역시 이루어집니다. 이는 SKKF mechanism에 의해 이루어지는데,  
MIRNet의 핵심이라고 할 수 있습니다. 이러한 방법의 차이는 최종적으로, 여러 scale에서 얻어낸 feature들을 기존의 방식처럼 단순하게 더하거나, 평균을 내거나 합치는(concatenation) 방식이 아니라 feature들을 동적으로 선택하기 때문에, 더 좋은 결과를 얻어낼 수 있습니다.

### Proposed Method
![mirnet_fig1](/assets/img/mirnet/fig1.png)
<figcaption>fig1. MIRNet의 residual recursive 구조 </figcaption>

#### 1. RRG(Recursive Residual Group)

&nbsp;&nbsp; RRG는 Recursive Residual Group의 약자로, 연속되는 MRB 구조를 포함하고 있습니다. MRB 구조의 재귀로 성능 향상을 위해 사용됩니다. 논문에서는 2개의 RRG를 사용하는 것을 제안하고 있습니다

#### 2. MRB(Multiscale Residual Block)

&nbsp;&nbsp; MIRNet 구조의 핵심이라고 할 수 있는 Block입니다. 하나의 RRG 안에 여러개의 MRB가 포함되어 있는 구조이며, 논문에서는 하나의 RRG에 3개의 MRB를 넣어 예시로 사용하였습니다. MRB는 다양한 크기의 공간 정보를 효과적으로 다루기 위해 원본 크기의 high-resolution representations 외에도 low-resolution에서도 feature들을 뽑아내어 풍부한 contextual information을 얻어냅니다. MRB는 여러개의 fully-convolutional stream들이 평행으로 이어져 있으며, 이는 각 stream들의 low-resolution feature와 high-resolution feature 간의 정보 교환을 가능하게 합니다.

#### 3. SKFF(Selective Kernel Feature Fusion)

&nbsp; &nbsp; MRB의 구조를 이루고 있는 요소 중 하나인 SKFF는 Feature를
#### 4. DAU(Dual Attention Unit)
#### 5. Residual Resizing Module
