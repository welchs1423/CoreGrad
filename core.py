import numpy as np
import random


class Tensor:

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t**2) * out.grad

        out._backward = _backward
        return out

    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
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
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)

        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self._op}')"


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin):
        self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Tensor(0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
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
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
class SGD:
    def __init__(self, parameters, lr=0.01):
        # 학습할 모델의 파라미터 리스트와 학습률을 초기화합니다
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        # 파라미터의 기울기를 0으로 초기화합니다
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        # 역전파로 계산된 기울기를 사용하여 파라미터를 업데이트합니다
        for p in self.parameters:
            p.data -= self.lr * p.grad


if __name__ == '__main__':
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])

    epochs = 20
    learning_rate = 0.05

    # Optimizer 객체를 생성하고 모델의 파라미터를 등록합니다
    optimizer = SGD(model.parameters(), lr=learning_rate)

    print(f"--- 학습 시작 (Total Params: {len(model.parameters())}) ---")

    for k in range(epochs):
        ypred = [model(x) for x in xs]
        loss = sum(((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)), 0.0)

        # 기존 모델의 zero_grad 대신 Optimizer의 zero_grad를 호출합니다
        optimizer.zero_grad()
        loss.backward()

        # 직접 파라미터를 수정하던 루프를 지우고 step 메서드 하나로 처리합니다
        optimizer.step()

        if k % 2 == 0:
            print(f"Epoch {k} | Loss: {loss.data.item():.4f}")

    final_pred = [model(x).data.item() for x in xs]
    print("-" * 30)
    print(f"Final Predictions: {[round(p, 4) for p in final_pred]}")
    print(f"Target Values:     {ys}")