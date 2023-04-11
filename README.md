## 요약
본 논문에서는 합성곱 신경망(Convolutional Neural Networks, CNN)과 탄성왜곡(Elastic Distortion) 기법을 통한 데이터 증강 기법을 활용하여 학습 데이터를 구축하는 프레임워크를 제안한다. 실제 균열 이미지는 정형화된 형태가 없고 복잡한 패턴을 지니고 있어 구하기 어려울 뿐만 아니라, 데이터를 확보할 때 위험한 상황에 노출될 우려가 있다. 이러한 데이터베이스 구축 문제점을 본 논문에서 제안하는 데이터 증강 기법을 통해 비용적, 시간적 측면에서 효율적으로 해결한다. 세부적으로는 DeepCrack의 데이터를 10배 이상 증가하여 실제 균열의 특징을 반영한 메타 데이터를 생성하여 U-net을 학습하였다. 성능을 검증하기 위해 균열 탐지 연구를 진행한 결과, IoU 정확도가 향상되었음을 확인하였다. 데이터를 증강하지 않았을 경우 잘못 예측(FP)된 경우의 비율이 약 25%였으나, 데이터 증강을 통해 3%까지 감소하였음을 확인하였다.

### 설명
최근 국내외에서 빈번히 발생하는 건축물 붕괴 사건을 CS(Computer Science) 관점에서 해결하고자 한다. 균열은 작업자가 현장에서 직접 눈으로 찾을 때 위험 사고에 노출될 우려가 있고 많은 시간과 비용이 소모되며 작업자의 시야, 날씨와 같은 외부 변수에 의해 영향을 받아 정확도에 한계가 있다. 본 논문에서는 데이터 세트 확장을 통해 인공신경망의 성능을 향상하였으며, 이 과정에서 균열의 방향과 두께를 고려한 인공신경망 기반 데이터 증강 기법을 활용하여 사용자의 직접적인 수집 없이도 균열 데이터를 대량으로 확보하였다.
이를 구현하기 위한 기여도는 다음과 같다.

● CNN과 탄성왜곡 기법을 통해 실제 균열 패턴과  형태를 반영한 새로운 균열 이미지를 생성하는 특징 전달 네트워크 개발        
● 다양한 균열에서 U-NET을 통한 균열 감지
### 결과

![image](https://user-images.githubusercontent.com/76280200/231085669-9e397945-68cb-4e1e-a6b4-3f38d63ca347.png)         
Fig 8의 (a)는 Averaging Blur를 적용한 실제 균열 이미지이며, (b)는 증강하기 전 검출 결과, (c)는 데이터 증강 후 검출 결과이다.    
데이터 증강한 모델로 훈련한 결과에서 (b) 상단에서 나타난 노이즈가 감소하고 정확도가 향상된 점을 확인하였다.
 
### 결론
본 연구에서는 데이터 증강 기법을 통해 데이터를 확장하고 균열 감지 테스트를 진행하여 결과를 WEB으로 시각화함으로써 시스템 상용화를 도모하고자 하였다. 작업자의 수작업 없이 인공신경망을 활용한 데이터 증강을 통해 대량의 균열 데이터를 확보할 수 있으며 기존 육안 점검을 통한 인력 중심의 비효율적 요소들을 인공신경망을 통해 자동화하였다. 데이터 확장을 통해 균열 감지 성능이 향상하였음을 확인하였다. 이는 콘크리트 균열에 제한적으로 활용되는 것이 아닌 균열과 같은 가지 형태의 패턴에 다양하게 응용할 수 있다. 향후 연구에서는 다른 딥러닝 모델을 활용하여 모델 간의 성능을 비교 평가하여 최적의 모델을 탐색하고자 한다.

***
학부연구
(2022/01 ~ 2023/02)
