# 머신러닝 & 딥러닝 시험 문제

## 1. 객관식 문제

### 1-1. 머신러닝의 3요소가 아닌 것은?
1. 데이터 기반 학습
2. 패턴 인식
3. 자율적 개선
4. 수동적 프로그래밍

### 1-2. 다음 중 Supervised Learning의 특징으로 올바른 것은?
1. Label이 없는 데이터로 학습
2. 정답 데이터와 예측값의 차이를 에러로 계산
3. 패턴과 관계를 찾아내는 학습
4. 보상을 최대화하는 학습

### 1-3. L1 Regularization의 특징은?
1. 가중치 제곱 합에 패널티 부여
2. 가중치 절댓값 합에 패널티 부여
3. 모든 가중치를 균등하게 감소
4. Overfitting을 유발

### 1-4. ReLU 활성화 함수의 동작 방식은?
1. 0과 1 사이의 값으로 변환
2. 0이하는 0, 0초과는 그대로 출력
3. 모든 값을 1 또는 0으로 변환
4. 모든 출력값의 합이 1이 되게 변환

### 1-5. Backpropagation에서 사용하는 수학적 규칙은?
1. Taylor Series
2. Chain Rule
3. Fourier Transform
4. Newton's Method

---

## 2. OX 문제

### 2-1. Overfitting은 학습 데이터에서는 성능이 좋지만 새로운 데이터에서는 성능이 떨어지는 현상이다. (O/X)

### 2-2. Semi-supervised Learning은 Label이 있는 데이터만 사용한다. (O/X)

### 2-3. Sigmoid 함수는 다중 분류에 주로 사용된다. (O/X)

### 2-4. Recall과 Precision은 trade-off 관계이다. (O/X)

### 2-5. 단층 퍼셉트론(SLP)은 XOR 연산을 해결할 수 있다. (O/X)

---

## 3. 빈칸 채우기

### 3-1. Perceptron의 Linear Function 공식
```
y = [빈칸1] + [빈칸2]
```

### 3-2. 강화학습의 4가지 구성요소
- [빈칸1]: 학습하고 행동하는 주체
- [빈칸2]: 에이전트가 상호작용하는 세계
- [빈칸3]: 특정 행동에 대한 긍정적/부정적 피드백
- [빈칸4]: 에이전트가 특정 상태에서 취하는 선택

### 3-3. Confusion Matrix 평가지표
```
Accuracy = ([빈칸1] + [빈칸2]) / 전체
Precision = [빈칸3] / (TP + FP)
Recall = TP / (TP + [빈칸4])
```

### 3-4. 딥러닝 훈련 원리 4단계
1. [빈칸1] Propagation (순전파)
2. Loss [빈칸2] (손실 계산)
3. [빈칸3] (역전파)
4. [빈칸4] Update (가중치 업데이트)

---

## 4. 단답형 문제

### 4-1. Label이 있는 일부 데이터(약 20%)와 Label이 없는 대량의 데이터로 학습하는 방법은?

### 4-2. 가중치가 커지는 것을 방지하기 위해 손실 함수에 가중치의 합을 더하는 기법은?

### 4-3. 출력층에서 다중 분류 시 사용되며, 모든 출력값의 합이 1이 되게 만드는 활성화 함수는?

### 4-4. 정답 데이터와 예측 데이터 사이의 차이를 의미하는 용어는?

### 4-5. 1969년 Marvin L. Minsky가 단층 퍼셉트론이 해결하지 못한다고 증명한 연산은?

---

## 5. 서술형 문제

### 5-1. Overfitting이 발생하는 이유와 해결 방법 2가지를 설명하시오.

### 5-2. Weight(가중치)와 Bias(편향)의 역할을 각각 설명하시오.

### 5-3. L1 Regularization과 L2 Regularization의 차이점을 설명하시오.

### 5-4. Supervised Learning과 Reinforcement Learning의 목적을 비교하여 설명하시오.

### 5-5. Forward Propagation과 Backpropagation의 역할을 각각 설명하시오.

---

## 6. 계산 문제

### 6-1. 다음 Confusion Matrix를 보고 물음에 답하시오.

|  | 실제 정답 | 실제 오답 |
|---|---|---|
| 예측 정답 | 40 | 10 |
| 예측 오답 | 5 | 45 |

1. Accuracy를 계산하시오.
2. Precision을 계산하시오.
3. Recall을 계산하시오.
4. F1-Score를 계산하시오.

### 6-2. Error = 정답 - 예측일 때, 정답이 100이고 예측이 85라면 Error는?

---

## 7. 매칭 문제

### 7-1. 다음 활성화 함수와 특징을 연결하시오.

**활성화 함수:**
- A. Step Function
- B. Sigmoid
- C. ReLU
- D. Softmax

**특징:**
1. 0과 1 사이의 부드러운 확률 표현, 이진 분류
2. 1 또는 0, 계단형 그래프
3. 마이너스면 0, 플러스면 그대로
4. 다중 분류, 모든 출력값 합이 1

### 7-2. 학습 방법과 사용 사례를 연결하시오.

**학습 방법:**
- A. Supervised Learning
- B. Unsupervised Learning
- C. Reinforcement Learning

**사용 사례:**
1. 자율주행, 게임 AI
2. 이미지 분류, 회귀 분석
3. 뉴스 주제 분류, 이상거래 탐지

---

## 8. 용어 설명

### 8-1. 다음 용어를 설명하시오.

1. **Generalization Ability (일반화 능력)**
2. **Vanishing Gradient (기울기 소실)**
3. **GIGO (Garbage In, Garbage Out)**
4. **Hyperplane (초평면)**
5. **Feature Extraction (특징 추출)**

---

## 9. 역사 문제

### 9-1. 다음 딥러닝 역사의 빈칸을 채우시오.

- 1958년: [빈칸1]이 단층 퍼셉트론(SLP) 구현
- 1974년: [빈칸2] 알고리즘 논문 등장
- 1986년: [빈칸3]이 Backpropagation으로 MLP 학습 증명
- 2010년: Geoffrey Hinton이 학습에 [빈칸4] 사용
- 2017년: Google DeepMind 팀 [빈칸5] 논문 발표

---

## 10. 응용 문제

### 10-1. 의료 분야에서 암 진단 AI를 개발한다고 가정하자. Precision과 Recall 중 어느 지표가 더 중요한지 설명하고 그 이유를 서술하시오.

### 10-2. 모델이 학습 데이터를 지나치게 외워서 일반화 능력이 떨어졌다. 이 문제를 해결하기 위한 방법 3가지를 제시하시오.

### 10-3. Biological Neuron의 구성요소(Dendrite, Axon, Synapse)와 인공신경망의 요소를 매칭하여 설명하시오.
