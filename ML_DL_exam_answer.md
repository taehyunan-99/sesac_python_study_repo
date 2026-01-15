# 머신러닝 & 딥러닝 시험 정답

## 1. 객관식 문제 정답

### 1-1. 정답: 4번
- 머신러닝의 3요소: 데이터 기반 학습, 패턴 인식, 자율적 개선

### 1-2. 정답: 2번
- Supervised Learning은 정답(Label)이 있는 데이터로 학습하며, 에러(Loss)는 모델의 예측값과 정답 데이터의 차이

### 1-3. 정답: 2번
- L1 Regularization(Lasso): 가중치의 절댓값 합에 비례하는 패널티를 손실함수에 추가
- L2 Regularization(Ridge): 가중치의 제곱 합에 패널티

### 1-4. 정답: 2번
- ReLU: 0이하는 0, 0초과 시 기울기가 1인 직선 (마이너스면 0, 플러스면 그대로)

### 1-5. 정답: 2번
- Backpropagation은 Chain Rule(합성함수의 미분을 계산하는 수학적 규칙) 사용

---

## 2. OX 문제 정답

### 2-1. O
- Overfitting: 학습 데이터에 대해서는 성능이 매우 좋지만, 새로운 데이터에서는 성능이 급격하게 떨어짐

### 2-2. X
- Semi-supervised Learning: Label이 있는 일부 데이터(약 20%)와 Label이 없는 대량의 데이터로 학습

### 2-3. X
- Sigmoid: 이진 분류에 사용
- Softmax: 다중 분류에 사용

### 2-4. O
- Recall과 Precision은 trade-off 관계

### 2-5. X
- 1969년 Marvin L. Minsky에 의해 단층 퍼셉트론은 XOR 연산을 못함이 증명됨

---

## 3. 빈칸 채우기 정답

### 3-1. Perceptron의 Linear Function 공식
```
y = Σ(wi*xi) + b
또는
y = w1x1 + w2x2 + ... + wnxn + b
```
- 빈칸1: Σ(wi*xi) 또는 가중합
- 빈칸2: b (bias, 편향)

### 3-2. 강화학습의 4가지 구성요소
- 빈칸1: 에이전트(Agent)
- 빈칸2: 환경(Environment)
- 빈칸3: 보상(Reward)
- 빈칸4: 행동(Action)

### 3-3. Confusion Matrix 평가지표
```
Accuracy = (TP + TN) / 전체
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```
- 빈칸1: TP
- 빈칸2: TN
- 빈칸3: TP
- 빈칸4: FN

### 3-4. 딥러닝 훈련 원리 4단계
1. Forward Propagation (순전파)
2. Loss Calculation (손실 계산)
3. Backpropagation (역전파)
4. Weight Update (가중치 업데이트)

---

## 4. 단답형 문제 정답

### 4-1.
**정답**: Semi-supervised Learning (준지도학습)

### 4-2.
**정답**: Regularization (정규화)

### 4-3.
**정답**: Softmax

### 4-4.
**정답**: Error (오차) 또는 Loss (손실)

### 4-5.
**정답**: XOR 연산

---

## 5. 서술형 문제 정답

### 5-1. Overfitting이 발생하는 이유와 해결 방법

**발생 이유**:
- 모델이 학습 데이터를 지나치게 외우는 현상
- 일반화 능력이 떨어짐

**해결 방법**:
1. **Regularization (정규화)**: 모델의 복잡도에 패널티를 부여하여 가중치가 커지는 것을 방지
2. **Feature Selection**: Feature 수를 줄여서 모델의 복잡도 감소 (단, Underfitting 조심)

### 5-2. Weight와 Bias의 역할

**Weight (가중치)**:
- 데이터의 중요도를 나타냄
- 입력값에 곱해져서 각 특징의 영향력을 조절

**Bias (편향)**:
- 데이터의 민감도를 나타냄
- 활성화 함수의 출력을 조정하여 모델의 유연성을 높임

### 5-3. L1과 L2 Regularization의 차이점

**L1 Regularization (Lasso)**:
- 가중치의 절댓값 합에 비례하는 패널티를 손실함수에 추가
- 일부 가중치를 정확히 0으로 만듦
- Feature Selection에 유용 (일부 특성 제거)

**L2 Regularization (Ridge)**:
- 가중치의 제곱 합에 비례하는 패널티를 손실함수에 추가
- 가중치 값이 전체적으로 너무 커지지 않도록 제한
- 모든 가중치를 0에 가깝게 만들지만 완전히 0으로는 만들지 않음

### 5-4. Supervised Learning과 Reinforcement Learning의 목적 비교

**Supervised Learning**:
- 목적: 모델의 에러 최소화
- 반복 학습을 통해 에러를 줄이며 모델을 개선

**Reinforcement Learning**:
- 목적: 에이전트의 보상 최대화
- 시행착오를 통해 보상을 최대화하는 최적의 행동을 학습
- 성능 평가: 에이전트의 행동 최적화

### 5-5. Forward Propagation과 Backpropagation의 역할

**Forward Propagation (순전파)**:
- 입력 데이터가 가중치와 활성 함수를 거쳐 은닉층을 통과
- 출력층에서 예측결과(Prediction)를 생성

**Backpropagation (역전파)**:
- 계산된 오차를 줄이기 위해, 오차를 신경망 역방향으로 이동
- 각 층의 가중치에 대한 오차 기울기를 계산 (Chain Rule 사용)
- 경사하강법으로 가중치를 업데이트하기 위한 기울기 제공

---

## 6. 계산 문제 정답

### 6-1. Confusion Matrix 계산

주어진 데이터:
- TP = 40, FP = 10
- FN = 5, TN = 45
- 전체 = 100

**1. Accuracy**
```
Accuracy = (TP + TN) / 전체
        = (40 + 45) / 100
        = 85 / 100
        = 0.85 (85%)
```

**2. Precision**
```
Precision = TP / (TP + FP)
         = 40 / (40 + 10)
         = 40 / 50
         = 0.8 (80%)
```

**3. Recall**
```
Recall = TP / (TP + FN)
      = 40 / (40 + 5)
      = 40 / 45
      = 0.889 (88.9%)
```

**4. F1-Score**
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
        = 2 × (0.8 × 0.889) / (0.8 + 0.889)
        = 2 × 0.711 / 1.689
        = 1.422 / 1.689
        = 0.842 (84.2%)
```

### 6-2. Error 계산

```
Error = 정답 - 예측
     = 100 - 85
     = 15
```

---

## 7. 매칭 문제 정답

### 7-1. 활성화 함수 매칭

- A. Step Function → 2번 (1 또는 0, 계단형 그래프)
- B. Sigmoid → 1번 (0과 1 사이의 부드러운 확률 표현, 이진 분류)
- C. ReLU → 3번 (마이너스면 0, 플러스면 그대로)
- D. Softmax → 4번 (다중 분류, 모든 출력값 합이 1)

### 7-2. 학습 방법과 사용 사례 매칭

- A. Supervised Learning → 2번 (이미지 분류, 회귀 분석)
- B. Unsupervised Learning → 3번 (뉴스 주제 분류, 이상거래 탐지)
- C. Reinforcement Learning → 1번 (자율주행, 게임 AI)

---

## 8. 용어 설명 정답

### 8-1. 용어 설명

**1. Generalization Ability (일반화 능력)**
- 학습된 모델이 데이터의 패턴을 파악하는 능력
- 머신러닝의 목적은 모델의 일반화 능력을 향상시키는 것

**2. Vanishing Gradient (기울기 소실)**
- 1990년대 딥러닝의 한계로 등장한 문제
- 역전파 과정에서 기울기가 점점 작아져서 학습이 제대로 되지 않는 현상

**3. GIGO (Garbage In, Garbage Out)**
- 쓰레기 데이터를 입력하면 쓰레기 결과가 나온다는 의미
- 머신러닝은 데이터 품질에 의존적이며, 데이터 수집, 정제, Labeling이 모델 설계보다 중요한 경우가 많음

**4. Hyperplane (초평면)**
- n-1 차원에 존재하는 평면
- 데이터를 분류하는 경계면

**5. Feature Extraction (특징 추출)**
- 복잡한 데이터에서 중요한 특징을 추출하는 과정
- 표준정규분포표로 가공하여 이해도를 높임

---

## 9. 역사 문제 정답

### 9-1. 딥러닝 역사

- 1958년: **Frank Rosenblatt**이 단층 퍼셉트론(SLP) 구현
- 1974년: **Backpropagation(역전파)** 알고리즘 논문 등장
- 1986년: **Geoffrey Hinton**이 Backpropagation으로 MLP 학습 증명
- 2010년: Geoffrey Hinton이 학습에 **ReLU** 사용
- 2017년: Google DeepMind 팀 **Attention Is All You Need** 논문 발표

---

## 10. 응용 문제 정답

### 10-1. 암 진단 AI에서 중요한 지표

**정답: Recall (재현율, 민감도)이 더 중요**

**이유**:
- Recall = TP / (TP + FN)
- FN (False Negative): 실제 암 환자를 정상으로 오판
- 암 진단에서 환자를 놓치는 것(FN)은 치명적
- Precision보다 Recall이 높아야 실제 암 환자를 최대한 많이 찾아낼 수 있음
- 오진(FP)보다 놓침(FN)이 더 위험

### 10-2. 일반화 능력 향상 방법 3가지

**1. Regularization (정규화)**
- L1 또는 L2 Regularization을 적용하여 가중치가 커지는 것을 방지

**2. Feature Selection**
- Feature 수를 줄여서 모델의 복잡도를 감소 (단, Underfitting 주의)

**3. 더 많은 학습 데이터**
- 다양한 데이터로 재학습하여 모델이 패턴을 일반화할 수 있도록 함

### 10-3. Biological Neuron과 인공신경망 매칭

**Dendrite (수상돌기) ↔ Input Layer (입력층)**
- 전기신호 입력을 받는 역할
- 인공신경망에서는 입력 데이터를 받음

**Axon (축삭돌기) ↔ 신호 전달 통로 / 연결**
- 신호를 전달하는 통로
- 인공신경망에서는 활성화 값을 다음 층으로 전달

**Synapse (시냅스) ↔ Weight (가중치)**
- 뉴런 간 연결 지점
- 시냅스 강도 = 가중치
- 시냅스 가소성 (자주 사용하는 부분 증폭) = 가중치 학습/업데이트
- 학습의 핵심 메커니즘
