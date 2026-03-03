import numpy as np


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
        self.grad = np.array([1.0], dtype=np.float32)

        # 3. 위상 정렬된 리스트를 거꾸로(reversed) 순회하며, 각 노드에 저장된 미분 규칙(_backward)을 차례대로 실행합니다!
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op='{self._op}')"


# =========================================
# 실행 테스트
# =========================================
if __name__ == '__main__':
    # 1. 신경망 모형 세팅: y = a*b + b*c
    a = Tensor([2.0])
    b = Tensor([3.0])
    c = Tensor([4.0])

    # 2. Forward (계산 실행)
    ab = a * b
    bc = b * c
    y = ab + bc

    # 3. Backward (자동 미분 실행)
    y.backward()

    # 4. 결과 출력
    print(f"최종 결과 y: {y.data}")
    print(f"dy/da: {a.grad}")  # 정답: 3.0 (b의 값)
    print(f"dy/db: {b.grad}")  # 정답: 6.0 (a + c의 값) -> b가 양쪽 연산에 모두 참여했으므로!
    print(f"dy/dc: {c.grad}")  # 정답: 3.0 (b의 값)