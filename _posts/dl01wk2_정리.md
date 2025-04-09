# 딥러닝 01wk2 이해하기

## 기본 세팅

```python
import torch
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (4.5, 3.0)
```

---

## 회귀 모형

### A. 아이스 아메리카노

- x 데이터, y 데이터가 존재한다면 산점도를 그려볼 수 있음

```python
plt.plot(x,y,'o')
```

---

### B. 가짜자료 만들기

- 방법 1:  
  \( y_i = w_0 + w_1 x_i + \epsilon_i = 2.5 + 4x_i + \epsilon_i, \quad i = 1,2,\dots,n \)

```python
torch.manual_seed(43052)
x,_ = torch.randn(100).sort()
eps = torch.randn(100)*0.5
y = x * 4 + 2.5 + eps
```

- x = 온도, y = 판매량, eps = 오차

- 방법 2:  
  \( \mathbf{y} = \mathbf{X} \mathbf{W} + \boldsymbol{\epsilon} \)

```python
X = torch.stack([torch.ones(100),x],axis=1)
W = torch.tensor([[2.5],[4.0]])
y = X@W + eps.reshape(100,1)
x = X[:,[1]]
```

- 여기서 구한 x와 y로 직선 그래프를 그릴 수 있음

---

### C. 추세선

- 추세선이란 주어진 x, y로 파라미터 w를 최대한 비슷하게 찾는 것
- 추세선을 그리는 행위 = \( (w_0, w_1) \) 을 선택하는 일

---

### D. 손실 함수

#### 예제1. \( (\hat{w}_0, \hat{w}_1) = (-5, 10) \)

```python
plt.plot(x,y,'o',label=r"observed data: $(x_i,y_i)$")
What = torch.tensor([[-5.0],[10.0]])
plt.plot(x,X@What,'--',label=r"estimated line: $(x_i,\hat{y}_i)$")
plt.legend()
```

- X의 shape = (100, 2), What의 shape = (2, 1)
- x에 대한 예측값을 산점도로 그려보는 코드
- 더 적당한 추세선을 판단하기 위해서는 loss 개념이 필요

---

### E. Loss

$$
\sum_{i=1}^n (y_i - \hat{y}_i)^2 = (y - \hat{y})^\top (y - \hat{y}) = (y - X \hat{W})^\top (y - X \hat{W})
$$

#### loss의 특징

- \( y_i \approx \hat{y}_i \) 일수록 loss 값이 작음
- \( y_i \approx \hat{y}_i \) 가 되도록 \( (\hat{w}_0, \hat{w}_1) \) 을 잘 선택하면 loss 값이 작음

#### loss 사용해보기

```python
torch.sum((y - X@What)**2)
```

---

## 파이토치를 이용한 반복추정

1. 아무 점선이나 그리기  
2. 1단계 점선보다 더 좋은 점선으로 바꾸기  
3. 반복하기

---

### A. 1단계: 최초 점선 그리기

```python
What = torch.tensor([[-5.0],[10.0]])
yhat = X@What
```

---

### B. 2단계: loss 이용하기

```python
loss = torch.sum((y - yhat)**2)
```

- 이 loss를 줄이는 게 핵심
- 목표: \( \text{loss}(\hat{w}_0, \hat{w}_1) \) 를 최소로 하는 \( (\hat{w}_0, \hat{w}_1) \) 구하기
- 경사하강법 사용

---

### C. 3단계: 경사하강법 아이디어

#### (1차원)

1. 임의의 점을 찍는다  
2. 그 점에서 순간기울기를 구한다 (접선) → 미분  
3. 순간기울기의 부호를 보고 **반대방향으로 이동**

> 팁: 기울기의 크기에 비례하여 움직이는 정도 조절 → \( \alpha \) 도입

> 최종 수식:  
> \( \hat{w} \leftarrow \hat{w} - \alpha \cdot \frac{\partial}{\partial w} \text{loss}(w) \)

---

#### (2차원)

1. 임의의 점을 찍는다  
2. 그 점에서 순간기울기를 구한다 (접평면) → 편미분  
3. 기울기 부호에 반대방향으로 각각 이동

> 여기서도 \( \alpha \) 를 도입하여 이동량 조절

---

### 경사하강법이란?

- 손실 \( \text{loss} \) 를 줄이도록 \( \hat{\mathbf{W}} \) 를 개선하는 알고리즘  
- 업데이트 공식:  
  \( \text{새로운 W} = \text{이전 W} - \alpha \cdot \text{기울기} \)  
- 여기서 \( \alpha \) 는 보폭의 크기

---

### 구하고자 하는 것

$$
\hat{\mathbf{W}} = \arg\min_{\mathbf{W}} \text{loss}(\mathbf{W})
$$