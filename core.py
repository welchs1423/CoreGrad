import numpy as np
import random


class Tensor:

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        # 기울기(미분값)를 저장할 변수, 0으로 초기화
        self.grad = np.zeros_like(self.data)

        # 계산 그래프 구조
        self._prev = set(_children)
        self._op = _op

        # 나중에 미분될 때 부모 노드들에게 기울기를 어떻게 분배할지 정의하는 콜백 함수
        # 초기에는 아무것도 하지 않음 (leaf node)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # 덧셈: 내 기울기(out.grad)를 부모들에게 그대로(1.0) 더해준다.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # 곱셈: Chain Rule! 상대방의 값(data)에 내 기울기(out.grad)를 곱해서 더해준다.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # 뺄셈 (self - other)
    def __sub__(self, other):
        return self + (-other)

    # 부호 반전 (-self) -> 뺄셈 구현을 위해 필요합니다.
    def __neg__(self):
            return self * -1

    # 거듭제곱 (self ** other) -> MSE Loss 계산용
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "현재는 숫자 지수만 지원합니다."
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            # x**n 미분: n * x**(n-1) * out.grad
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # 🚀 대망의 자동 미분 기능
    def backward(self):
        # 1. 위상 정렬 (Topological Sort)
        # 그래프를 재귀적으로 탐색하면서, 모든 자식 노드가 먼저 추가된 후에 부모 노드가 추가되도록 순서를 보장합니다.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        # 최종 노드인 '나(self)'부터 역순 탐색 시작
        build_topo(self)

        # 2. 마지막 결과물의 기울기는 무조건 1.0부터 시작합니다 (de/de = 1)
        self.grad = np.ones_like(self.data)

        # 3. 위상 정렬된 리스트를 거꾸로(reversed) 순회하며, 각 노드에 저장된 미분 규칙(_backward)을 차례대로 실행합니다!
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self._op}')"

class Module:

    def zero_grad(self):
        # 학습 전, 모든 파라미터의 기울기를 0으로 초기화합니다.
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        # 상속받는 클래스에서 구현할 예정입니다.
        return []

class Neuron(Module):

    def __init__(self, nin):
        # nin: 입력의 개수 (number of inputs)
        # 가중치(w)를 랜덤하게 초기화합니다.
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        # 편향(b)은 0으로 초기화합니다.
        self.b = Tensor(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()  # 여기서 꺾어줍니다!

    def parameters(self):
        # 이 뉴런이 가진 학습 가능한 변수(w, b)들을 반환합니다.
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout):
        # nin: input dimension
        # nout: number of neurons in this layer
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

class MLP(Module):
    def __init__(self, nin, nouts):
        # nin: input dimension
        # nouts: list of output dimensions for each layer
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


if __name__ == '__main__':
    # 1. 간단한 데이터셋 구성 (입력 X, 정답 y)
    # 입력 데이터: 4개의 샘플 (각 3차원)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # 각 샘플의 정답(Target)

    # 2. 모델 초기화 (3입력 -> 4은닉 -> 4은닉 -> 1출력)
    model = MLP(3, [4, 4, 1])

    # 3. 학습 루프 (Training Loop)
    epochs = 20
    learning_rate = 0.01

    print(f"--- 학습 시작 (Total Params: {len(model.parameters())}) ---")

    for k in range(epochs):
        # 3-1. Forward Pass: 모델을 통해 예측값 계산
        ypred = [model(x) for x in xs]

        # 3-2. Loss Function: 평균 제곱 오차 (MSE) 계산
        # Loss = sum((yout - ygt)^2)
        # 시작값을 0.0(float)으로만 지정해줘도 파이썬이 자동으로 Tensor 연산으로 유도합니다.
        loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), 0.0)

        # 3-3. Zero Grad: 이전 단계의 기울기 초기화
        model.zero_grad()

        # 3-4. Backward Pass: 역전파 실행
        loss.backward()

        # 3-5. Update (SGD): 기울기 반대 방향으로 파라미터 이동
        for p in model.parameters():
            p.data += -learning_rate * p.grad

        if k % 2 == 0:
            print(f"Epoch {k} | Loss: {loss.data.item():.4f}")

    # 4. 학습 후 결과 확인
    final_pred = [model(x).data.item() for x in xs]
    print("-" * 30)
    print(f"Final Predictions: {[round(p, 4) for p in final_pred]}")
    print(f"Target Values:     {ys}")