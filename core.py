import numpy as np
import random


class Tensor:
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

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # 곱셈: Chain Rule! 상대방의 값(data)에 내 기울기(out.grad)를 곱해서 더해준다.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

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
        # x: 입력 리스트 [x1, x2, ...]
        # w*x + b 연산을 수행합니다.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # 나중에 비선형성을 위해 .relu()를 붙일 수 있습니다. (지금은 일단 선형으로!)
        return act

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
    # Input data
    x = [Tensor(2.0), Tensor(3.0), Tensor(-1.0)]

    # MLP architecture: 3 inputs -> [4 hidden -> 4 hidden -> 1 output]
    nouts = [4, 4, 1]
    model = MLP(3, nouts)

    # Forward Pass
    out = model(x)

    # Backward Pass
    out.backward()

    print(f"MLP output result: {out.data}")
    print(f"Total trainable parameters in this MLP: {len(model.parameters())}")