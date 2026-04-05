# 🧠 CoreGrad

파이썬(Python)과 NumPy만으로 밑바닥부터 구현하는 딥러닝 자동 미분(Autograd) 엔진입니다. 거대 프레임워크의 도움 없이 순수 수학과 로직으로 텐서(Tensor) 연산 코어를 구축합니다.

---

### 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Library:** NumPy (다차원 배열 연산 및 행렬 계산)
* **Core Concepts:** * **Computational Graph:** 연산 과정을 그래프 형태로 추적
    * **Reverse-mode Autograd:** 역전파를 통한 자동 미분 구현
    * **Topological Sort:** 위상 정렬을 활용한 정확한 연산 순서 제어

---

### 🚀 진행 상황
* [x] Python & NumPy 기반 텐서 연산 코어 구축
* [x] 위상 정렬 기반 자동 미분(Autograd) 엔진 완성
* [x] Neuron/Layer/MLP 신경망 아키텍처 설계
* [x] ReLU 활성화 함수 및 MSE 손실 함수 구현
* [x] **학습률 최적화를 통한 실제 데이터 학습 루프 검증 성공**
* [x] 평균 제곱 오차(MSE) 손실 함수 구현
* [x] 확률적 경사 하강법(SGD) 최적화 알고리즘 적용
* [x] **전체 학습 루프(Forward-Loss-Backward-Update) 파이프라인 완성 및 검증**
* [x] 다중 레이어를 적층한 다층 퍼셉트론(MLP) 모델 클래스 구현
* [x] 다수의 뉴런을 묶어 병렬 처리하는 `Layer` 클래스 (nn.Linear 역할) 구현
* [x] Python 3.11 및 NumPy 기반 환경 구축
* [x] 연산 그래프 추적 가능한 `Tensor` 객체 설계
* [x] 자동 미분(Autograd) 엔진: 위상 정렬 및 역전파 구현 완료
* [x] 복합 연산 그래프(y = ab + bc) 기울기 검증 성공
* [x] **신경망 기본 단위인 `Module` 및 `Neuron` 클래스 구현 완료 (가중치 및 편향 자동 미분 테스트 통과)**
* [x] 음수 데이터 처리 및 비선형성 강화를 위한 Tanh 활성화 함수 구현
---

### 💻 실행 예시 (y = ab + bc)
```python
from core import Tensor

# 텐서 생성
a = Tensor([2.0])
b = Tensor([3.0])
c = Tensor([4.0])

# 계산 그래프 구성 (Forward)
y = (a * b) + (b * c)  # 18.0

# 자동 미분 실행 (Backward)
y.backward()

# 기울기(Gradient) 확인
print(a.grad) # [3.]
print(b.grad) # [6.] (a + c)
print(c.grad) # [3.]