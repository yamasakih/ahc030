#! /root/.pyenv/versions/3.11.4/bin/python
import sys
from pprint import pprint
from random import gauss
from functools import partial


def debug(*args, end="\n") -> None:  # type: ignore
    print(*args, end=end, file=sys.stderr)


dpprint = partial(pprint, stream=sys.stderr)

JUDGE = "\033[34m[Judge]\033[0m"
N: int
M: int
eps: float

D: list[list[int]] = []
V: list[list[int]] = []
E: list[float] = []

lines: list[str]
oils: set[tuple[int, int]] = set()
cost: int = 0


def read_input(input_data: str) -> None:
    with open(input_data, "r") as f:
        global N, M, eps, lines
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        N_, M_, eps_ = lines[0].split()
        N, M, eps = int(N_), int(M_), float(eps_)
        for i in range(M + 1, 2 * M + 1):
            d = list(map(int, lines[i].split()))
            D.append(d)
        for i in range(2 * M + 1, 2 * M + N + 1):
            v = list(map(int, lines[i].split()))
            V.append(v)
        for i in range(2 * M + N + 1, 2 * M + N + N**2 * 2 + 1):
            E.append(float(lines[i]))
        for i in range(N):
            for j in range(N):
                if V[i][j] > 0:
                    oils.add((i, j))


def is_unique(d: list[int], count: int) -> None:
    check_oils = set()
    for i in range(0, len(d), 2):
        x, y = d[i], d[i + 1]
        check_oils.add((x, y))
        # debug(f"{JUDGE} {len(check_oils)=}")
    assert len(check_oils) == count


def predict(vs: int, k: int) -> int:
    """
    パラメータepsを用いて、以下の平均 μ と分散 σ^{2} の正規分布からサンプルされた値をxとする。
    このとき、情報として得られる値は max(0,round(x)) である。
    平均 μ=(k−v(S))ϵ+v(S)(1−ϵ) 分散 σ2 = kϵ(1−ϵ)
    """
    mu = (k - vs) * eps + vs * (1 - eps)
    sigma = (k * eps * (1 - eps)) ** 0.5
    x = gauss(mu, sigma)
    debug(f"{JUDGE} {mu=}, {sigma=:0.4f}, {x=:0.4f}")
    return max(0, round(x))


def main() -> None:
    global cost
    args = sys.argv
    input_data = args[1]

    debug(f"{input_data=}")
    read_input(input_data)

    debug(f"{N=}, {M=}, {eps=}")
    debug(f"{D=}")
    debug(f"V")
    dpprint(V)
    debug(f"{E[:10]=}")
    debug("-----------------------------------")

    print(lines[0])
    debug(f"{JUDGE} {lines[0]}")
    for i in range(1, M + 1):
        print(lines[i])
        debug(f"{JUDGE} {lines[i]}")

    while True:
        type_, *query_ = input().split()
        query: list[int] = list(map(int, query_))
        # debug(f"{JUDGE} {type_}, {query=}")
        match type_:
            case "q":
                k, *d = query
                assert k >= 1
                if k == 1:
                    # 座標 (i, j) における油田埋蔵量 v を返す
                    cost += 1
                    assert len(d) == 2
                    i, j = d
                    print(V[i][j])
                    debug(f"{JUDGE} {V[i][j]}")
                else:
                    # 指定された座標の集合 S の油田埋蔵量の総和 v(S) の近似値を返す
                    is_unique(d, k)
                    cost += 1 / k**0.5
                    debug(f"{JUDGE} {cost=}")
                    sum_v = 0
                    for i in range(0, len(d), 2):
                        x, y = d[i], d[i + 1]
                        sum_v += V[x][y]
                    x = predict(sum_v, k)
                    print(x)
                    debug(f"{JUDGE} {x}")
            case "a":
                # 与えられた情報と油田の場所が一致しているか判定し結果を返す
                cost += 1
                count, *d = query
                # debug(f"{JUDGE} {count=} {d=}")
                assert len(d) % 2 == 0
                assert len(d) // 2 == len(oils)
                # 与えられた油田の場所に同じ値が入ってないか判定する
                is_unique(d, count)
                for i in range(0, len(d), 2):
                    x, y = d[i], d[i + 1]
                    if (x, y) not in oils:
                        # 一致していないので 0 を返す
                        cost += 1
                        print(0)
                        debug(f"{JUDGE} {0}")
                        break
                else:
                    # すべて一致しているので 1 を返し、プログラムの実行を終了する
                    print(1)
                    debug(f"{JUDGE} {1}")
                    debug(f"{JUDGE} {cost=}")
                    exit()


if __name__ == "__main__":
    main()
