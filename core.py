import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        # 1. 실제 데이터 (NumPy 배열로 래핑하여 고속 연산 준비)
        self.data = np.array(data, dtype=np.float32)

        # 2. 기울기(미분값)를 저장할 변수 (초기값은 0)
        self.grad = np.zeros_like(self.data)

        # 3.계산 그래프(Computational Graph)를 그리기 위한 핵심 요소
        # 자신이 어떤 텐서들을 통해(_prev), 어떤 연산(_op)으로 만들어졌는지 기억함
        self._prev = set(_children)
        self._op = _op

    # 파이썬 매직 메서드: a + b를 할 때 자동으로 호출됩니다.
    def __add__(self, other):
        # 숫자가 들어오면 Tensor 객체로 감싸줌.
        other = other if isinstance(other, Tensor) else Tensor(other)

        # 두 데이터를 더한 새로운 Tensor를 만들되, '자신'과 'other'를 부모로 기록
        out = Tensor(self.data + other.data, (self, other), '+')
        return out
    # 객체를 출력할때 예쁘게 보여주기 위한 메서든
    def __repr__(self):
        return f"Tensor(data={self.data}, op='{self._op}')"


if __name__ == '__main__':
    # 텐서 a와 b 생성
    a = Tensor([2.0])
    b = Tensor([3.0])

    # 덧셈 연산 발생! (이 순간 __add__ 가 호출됨)
    c = a + b

    print("a =", a)
    print("b =", b)
    print("c (a + b) =", c)
    print("-" * 30)
    print("c가 기억하는 부모 노드들 (어떻게 만들어졌는가?):")
    for parent in c._prev:
        print(" ->", parent)