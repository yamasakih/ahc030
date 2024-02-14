from dataclasses import dataclass, field
import itertools
import sys
from typing import Any, Final, Optional, Union
from random import gauss, randint
from pprint import pprint
from functools import partial
from collections import deque
import heapq


def debug(*args, end="\n") -> None:  # type: ignore
    print(*args, end=end, file=sys.stderr)


dpprint = partial(pprint, stream=sys.stderr)


AHC30: Final[str] = "\033[32m[AHC30]\033[0m"

# prior information
N: int
M: int
eps: float
oils: list[list[tuple[int, int]]] = []
oil_ends: list[tuple[int, int]] = []
P: list[list[float]]  # 油田がある確率
C: list[list[float]]  # 何回占ったか
D: list[list[int]]  # 掘って実際に調べた油田の量
G: list[list[list[int]]]  # 各油田のパーツ同士の結合情報
S: list[int]  # 各油田の面積


def debug_p(to_int: Union[float, list[int]] = -1) -> None:
    if type(to_int) != list:
        if to_int == -1:
            for p in P:
                debug(f"{AHC30} {' '.join(format(pi, '.2f') for pi in p)}")
        else:
            for p in P:
                debug(f"{AHC30} {' '.join('0' if pi < to_int else '1' for pi in p)}")  # type: ignore
    else:
        Q: list[list[Union[int, str]]] = [["-1"] * N for _ in range(N)]
        for i, j in enumerate(to_int):
            for x in range(N):
                for y in range(N):
                    if Q[x][y] != "-1":
                        continue
                    if P[x][y] <= j:
                        Q[x][y] = str(i)
        debug(f"{Q=}")
        for q in Q:
            debug(f"{AHC30} {' '.join(qi for qi in q)}")  # type: ignore


def debug_d() -> None:
    for i in range(N):
        line = ""
        for j in range(N):
            if D[i][j] == -1:
                line += "-1 "
            elif D[i][j] == 0:
                line += f" \033[32m{D[i][j]}\033[0m "
            else:
                line += f" \033[34m{D[i][j]}\033[0m "
        debug(f"{AHC30} {line}")


def estimate(x: int, y: int, i: int) -> tuple[float, int]:
    def _calculate_score(dx: int, dy: int) -> float:
        score = 0.0
        for t in range(len(oil)):
            nx = oil[t][0] + dx
            ny = oil[t][1] + dy
            if x + dx >= N or y + dy >= N:
                return -1
            if D[nx][ny] != -1:
                score += D[nx][ny]
            else:
                score += P[nx][ny]
        return score / len(oil)

    # 点 x, y を含めて i 番目の油田の形と推定し、その油田の確率に基づくスコアを返す
    oil = oils[i]
    # 点 oil[j] を x, y とした時のスコアを計算する
    best_score = -1.0
    best_idx = -1
    for s in range(len(oil)):
        dx = oil[s][0] - x
        dy = oil[s][1] - y
        score = _calculate_score(dx, dy)
        if score > best_score:
            best_score = score
            best_idx = s
    return best_score, best_idx


def play_all_dig() -> None:
    # drill every square
    has_oil = []
    for i in range(N):
        for j in range(N):
            print(f"q 1 {i} {j}")
            # debug(f"{AHC30} q 1 {i} {j}")  # ! comment when submittion
            resp = input()
            if resp != "0":
                has_oil.append((i, j))

    oil_positions = " ".join(map(lambda x: f"{x[0]} {x[1]}", has_oil))
    print(f"a {len(has_oil)} {oil_positions}")
    # debug(f"{AHC30} a {len(has_oil)} {oil_positions}")  # ! comment when submittion
    judge = int(input())
    # debug(f"{AHC30} {judge=}")  # ! comment when submittion
    assert judge == 1


def play_all_dig_with_counting() -> None:
    """
    play_all_dig と同じく愚直に掘っていくが総油田量が与えられた油田の面積と等しくなったら掘るのをやめる
    """
    oil_value = sum(S)
    has_oil = []
    for i in range(N):
        for j in range(N):
            print(f"q 1 {i} {j}")
            # debug(f"{AHC30} q 1 {i} {j}")  # ! comment when submittion
            resp = int(input())
            if resp != 0:
                has_oil.append((i, j))
                oil_value -= resp
            if oil_value == 0:
                break
        if oil_value == 0:
            break
    oil_positions = " ".join(map(lambda x: f"{x[0]} {x[1]}", has_oil))
    print(f"a {len(has_oil)} {oil_positions}")
    # debug(f"{AHC30} a {len(has_oil)} {oil_positions}")  # ! comment when submittion
    judge = int(input())
    # debug(f"{AHC30} {judge=}")  # ! comment when submittion
    assert judge == 1


@dataclass
class Area:
    x: list[int] = field(default_factory=list)
    y: list[int] = field(default_factory=list)

    def scry(self) -> None:
        fields = []
        for x, y in zip(self.x, self.y):
            fields.append(f"{x} {y}")
        command = f"q {len(fields)} {' '.join(fields)}"
        print(command)
        # debug(f"{AHC30} {command}")  # ! comment when submittion
        # 結果を P に反映する
        result = int(input())
        oil_value = result / len(fields)
        for x, y in zip(self.x, self.y):
            P[x][y] = (P[x][y] * C[x][y] + oil_value) / (C[x][y] + 1)
            C[x][y] += 1


@dataclass
class State:
    oils: list[tuple[int, int]] = field(default_factory=list)
    overlapped_oils: list[list[int]] = field(
        default_factory=lambda: [[0] * N for _ in range(N)]
    )
    scried_oils: list[list[int]] = field(
        default_factory=lambda: [[0] * N for _ in range(N)]
    )

    def dist(self) -> float:
        # P との距離を求める
        ret = 0
        for i in range(N):
            for j in range(N):
                ret += abs(P[i][j] - self.scried_oils[i][j])
        return ret

    def answer(self) -> None:
        oil_positions = []
        for i in range(N):
            for j in range(N):
                if self.overlapped_oils[i][j] != 0:
                    oil_positions.append(f"{i} {j}")
        # debug(  # ! comment when submittion
            # f"{AHC30} a {len(oil_positions)} {' '.join(oil_positions)}"  # ! comment when submittion
        # )  # ! comment when submittion
        print(f"a {len(oil_positions)} {' '.join(oil_positions)}")


def predict(vs: int, k: int) -> int:
    """
    パラメータepsを用いて、以下の平均 μ と分散 σ^{2} の正規分布からサンプルされた値をxとする。
    このとき、情報として得られる値は max(0,round(x)) である。
    平均 μ=(k−v(S))ϵ+v(S)(1−ϵ) 分散 σ2 = kϵ(1−ϵ)
    """
    mu = (k - vs) * eps + vs * (1 - eps)
    sigma = (k * eps * (1 - eps)) ** 0.5
    # x = mu + E[num_responses] * sigma
    x = gauss(mu, sigma)
    # debug(f"{AHC30} {vs=}, {mu=:0.4f}, {sigma=:0.4f}, {x=:0.4f}, {max(0, round(x))=}")
    return max(0, round(x))


def play_brute_force() -> None:
    # N と M が小さい場合にすべての油田をずらしながら配置して得られる情報を利用する

    def debug_b(d: int = 2) -> None:
        if d == 2:
            for i in range(N):
                line = ""
                for j in range(N):
                    if B[i][j] == 0:
                        line += f" \033[32m{B[i][j]:02}\033[0m "
                    else:
                        line += f" \033[34m{B[i][j]:02}\033[0m "
                debug(f"{AHC30} {line}")
        elif d == 4:
            for i in range(N):
                line = ""
                for j in range(N):
                    if B[i][j] == 0:
                        line += f" \033[32m{B[i][j]:04}\033[0m "
                    else:
                        line += f" \033[34m{B[i][j]:04}\033[0m "
                debug(f"{AHC30} {line}")

    global C, P, D, G
    P = [[0] * N for _ in range(N)]
    C = [[0] * N for _ in range(N)]
    D = [[-1] * N for _ in range(N)]
    B = [[0] * N for _ in range(N)]  # 油田が存在することがあるかどうか
    # すべての左上の座標を全探索し、角が一度も油田が存在し得ない場所を得る
    tmp = [[0] * N for _ in range(N)]
    for i in range(M):
        oil = oils[i]
        height, width = oil_ends[i]
        for sx in range(N - height):
            for sy in range(N - width):
                for dx, dy in oil:
                    x = sx + dx
                    y = sy + dy
                    # debug(f"{x=}, {y=}")
                    tmp[x][y] |= 1
        for i in range(N):
            for j in range(N):
                B[i][j] += tmp[i][j]
    # debug_b()  # ! comment when submittion
    # B の 油田が存在しないところを P, D に反映する
    for i in range(N):
        for j in range(N):
            if B[i][j] == 0:
                P[i][j] = 0
                D[i][j] = 0
    # debug_d()  # ! comment when submittion
    # (2, 2), (2, 5), ..., (5, 2), (5, 5), ... と格子状に掘っていく
    if eps > 0.05 or M >= 3:
        step = 2
    else:
        step = 3
    for i in range(2, N, step):
        for j in range(2, N, step):
            if D[i][j] != -1:
                continue
            print(f"q 1 {i} {j}")
            # debug(f"{AHC30} q 1 {i} {j}")  # ! comment when submittion
            result = int(input())
            D[i][j] = result
            P[i][j] = result
    # debug(f"{AHC30} ---------------------------")  # ! comment when submittion
    # debug_d()  # ! comment when submittion

    def is_ok(products: Any, B: Any) -> tuple[bool, Optional[State]]:
        B = [[0] * N for _ in range(N)]  # 重ね合わせた油田の配置
        for i, (sx, sy) in enumerate(products):
            oil = oils[i]
            for dx, dy in oil:
                x = sx + dx
                y = sy + dy
                if x >= N or y >= N:
                    return False, None
                if D[x][y] == 0:
                    return False, None
            # ok だったので B に反映する
            for dx, dy in oil:
                x = sx + dx
                y = sy + dy
                B[x][y] += 1
        state = State()
        state.overlapped_oils = B
        state.oils = products
        return True, state

    # すべてのマスを小さな四角形で一度ずつ占うようにする
    # 四角形を横長、縦長、正方形のいずれにするか調べる。縦 / 横 が 1.5 以上なら横長、逆なら縦長、それ以外なら正方形
    rectangle_counts = [0, 0, 0]  # 前から 0: 正方形, 1: 横長, 2: 縦長
    for m in range(M):
        height, width = oil_ends[m]
        if height / width > 1.5:
            rectangle_counts[2] += 1
        elif width / height > 1.5:
            rectangle_counts[1] += 1
        else:
            rectangle_counts[0] += 1
    # debug(f"{AHC30} {rectangle_counts=}")  # ! comment when submittion
    i = max((j, i) for i, j in enumerate(rectangle_counts))[1]
    scry_type = ""
    match i:
        case 0:  # 正方形
            scry_type = "square"
            # 3 x 3 の正方形で探索する
            if eps < 0.03:
                scry_oil_width = 5
                scry_oil_height = 5
            else:
                scry_oil_width = 3
                scry_oil_height = 3
        case 1:  # 横長
            scry_type = "yokonaga"
            # 2 x 3 の横長で探索する
            if eps < 0.03:
                scry_oil_width = 4
                scry_oil_height = 6
            else:
                scry_oil_width = 2
                scry_oil_height = 3
        case 2:  # 縦長
            scry_type = "tatenaga"
            # 3 x 2 の縦長で探索する
            if eps < 0.03:
                scry_oil_width = 6
                scry_oil_height = 4
            else:
                scry_oil_width = 3
                scry_oil_height = 2
    # debug(f"{AHC30} {i=} {scry_type=}")  # ! comment when submittion

    # N x N のマスを scry_oil で探索するため Area に分割する
    areas: list[Area] = []
    for i in range(0, N, scry_oil_height):
        for j in range(0, N, scry_oil_width):
            area = Area()
            for k in range(scry_oil_height):
                for l in range(scry_oil_width):
                    if i + k < N and j + l < N and D[i + k][j + l] == -1:
                        area.x.append(i + k)
                        area.y.append(j + l)
            if area.x:
                areas.append(area)
    # debug(f"{AHC30} {len(areas)=}")  # ! comment when submittion
    # debug(f"{AHC30} {areas=}")  # ! comment when submittion

    # すべてのエリアで占う
    for area in areas:
        area.scry()
    # debug_p()  # ! comment when submittion

    # 全探索し現在の P の形に似た油田の配置を探す。油田が多すぎる場合はその量に応じて適当に数カ所掘る
    while True:
        states = []
        # overlapped_oil_set = set([])  # TODO あとでハッシュで実装する
        for products_ in itertools.product(
            list(itertools.product(range(N), repeat=2)), repeat=M
        ):
            ok, state = is_ok(products_, B)
            if ok:
                states.append(state)
        # debug(f"{AHC30} {len(states)=}")  # ! comment when submittion
        # debug(f"{AHC30} {states[0]=}")  # ! comment when submittion
        # TODO よりキーとなるところを掘るように後で実装したい
        x = len(states)
        y = 1000
        if x > y:
            while x > y:
                i = randint(0, N - 1)
                j = randint(0, N - 1)
                if D[i][j] == -1:
                    print(f"q 1 {i} {j}")
                    # debug(f"{AHC30} q 1 {i} {j}")
                    result = int(input())
                    D[i][j] = result
                    P[i][j] = result
                    x -= y
                # debug_d()  # ! comment when submittion
        else:
            break

    # AHC30 自身で各 State に対して predict を a 回行いその結果の平均を scried_oils に保存する
    # a = 100
    a = 1000
    for state in states:
        for _ in range(a):
            for area in areas:
                vs = 0
                k = len(area.x)
                for x, y in zip(area.x, area.y):
                    vs += state.overlapped_oils[x][y]
                result = predict(vs, k)
                oil_value = result / k
                for x, y in zip(area.x, area.y):
                    state.scried_oils[x][y] += oil_value
        # 平均化する
        for i in range(N):
            for j in range(N):
                state.scried_oils[i][j] /= a
        # debug(f"{AHC30} {state.scried_oils=}")  # ! comment when submittion
        # dpprint(state.scried_oils, width=1000)  # ! comment when submittion
    # 各 State と P の間の距離を求める
    sorted_dist = sorted([(state.dist(), i) for i, state in enumerate(states)])
    sorted_idx = [i for _, i in sorted_dist]
    # debug(f"{AHC30} {sorted_dist[:3]=}")  # ! comment when submittion
    # debug(f"{AHC30} {sorted_idx=}")  # ! comment when submittion
    # debug(states[sorted_dist[0][1]])  # ! comment when submittion
    # 一番距離が近い State を answer として query を送る
    cur = 0
    while True:
        states[sorted_idx[cur]].answer()
        cur += 1
        judge = int(input())
        debug(f"{AHC30} {judge=}")  # ! comment when submittion
    # assert judge == 1
        if judge == 1:
            print("c")
            exit()
    exit()
    """
    for _ in range(100):
        # これまでの情報で存在することができるすべての組み合わせの油田の場所を調べる
        # ついでに OK な products のみで探索し一度も油田が存在し得ない場所を得る
        B = [[0] * N for _ in range(N)]  # 油田が存在することがあるかどうか
        products = []
        for products_ in itertools.product(
            list(itertools.product(range(N), repeat=2)), repeat=M
        ):
            ok, B_ = is_ok(products_, B)
            if ok:
                products.append(products_)
                B = B_
        # debug(f"{AHC30} {len(products)=}")  # ! comment when submittion
        # debug_b(4)  # ! comment when submittion
        # products の要素が１つだけになったら確定
        if len(products) == 1:
            product = products[0]
            # debug(f"{product=}")  # ! comment when submittion
            oil_positions = set([])
            for i, (sx, sy) in enumerate(product):
                oil = oils[i]
                for dx, dy in oil:
                    x = sx + dx
                    y = sy + dy
                    oil_positions.add(f"{x} {y}")
            # debug(  # ! comment when submittion
                # f"{AHC30} a {len(oil_positions)} {' '.join(oil_positions)}"  # ! comment when submittion
            # )  # ! comment when submittion
            print(f"a {len(oil_positions)} {' '.join(oil_positions)}")
            judge = int(input())
            # debug(f"{AHC30} {judge=}")  # ! comment when submittion
            assert judge == 1
            exit()
        # products の要素が 1 にしぼりきれないが D が先にすべて埋まる場合があるのでそれも確定
        if all(D[i][j] != -1 for i in range(N) for j in range(N)):
            debug(f"{AHC30} all D is filled")
            oil_positions = set([])
            for i in range(N):
                for j in range(N):
                    if D[i][j] != 0:
                        oil_positions.add(f"{i} {j}")
            # debug(  # ! comment when submittion
                # f"{AHC30} a {len(oil_positions)} {' '.join(oil_positions)}"  # ! comment when submittion
            # )  # ! comment when submittion
            print(f"a {len(oil_positions)} {' '.join(oil_positions)}")
            judge = int(input())
            # debug(f"{AHC30} {judge=}")  # ! comment when submittion
            assert judge == 1
            exit()
        # B の 油田が存在しないところを P, D に反映する
        for i in range(N):
            for j in range(N):
                if B[i][j] == 0:
                    P[i][j] = 0
                    D[i][j] = 0
        # debug_d()  # ! comment when submittion
        # B の確率が低いところから 5 個掘る
        q = []
        r = 5
        for i in range(N):
            for j in range(N):
                if D[i][j] != -1:
                    continue
                if B[i][j] == 0:
                    continue
                q.append((B[i][j], i, j))
        heapq.heapify(q)
        for _ in range(r):
            if not q:
                break
            _, best_i, best_j = heapq.heappop(q)
            print(f"q 1 {best_i} {best_j}")
            # debug(f"{AHC30} q 1 {best_i} {best_j}")  # ! comment when submittion
            result = int(input())
            D[best_i][best_j] = result
            P[best_i][best_j] = result
        # debug(f"{AHC30} ---------------------------")  # ! comment when submittion
        # debug_d()  # ! comment when submittion

    print("c")
    """

def play_random() -> None:
    global C, P, D, G
    P = [[0] * N for _ in range(N)]
    C = [[0] * N for _ in range(N)]
    D = [[-1] * N for _ in range(N)]
    # ランダムな場所でいずれかの油田の形で占う
    times = 50
    while times > 0:
        # ランダムに配置する油田のインデックスを決める
        i = randint(0, M - 1)
        oil = oils[i]
        height, width = oil_ends[i]
        # 左上の座標をランダムに求める
        sx = randint(0, N - 1 - height)
        sy = randint(0, N - 1 - width)
        # 占う
        times -= 1
        fields = []
        for dx, dy in oil:
            x = sx + dx
            y = sy + dy
            fields.append(f"{x} {y}")
        command = f"q {len(fields)} {' '.join(fields)}"
        print(command)
        # debug(f"{AHC30} {command}")  # ! comment when submittion
        # 結果を P に反映する
        result = int(input())
        oil_value = result / len(fields)
        for dx, dy in oil:
            x = sx + dx
            y = sy + dy
            P[x][y] = (P[x][y] * C[x][y] + oil_value) / (C[x][y] + 1)
            C[x][y] += 1

    # # ランダムな場所をピックアップして占う
    # times = 50
    # while times > 0:
    #     k = 9
    #     fields = set([])
    #     while k > 0:
    #         debug(f"{k=}")
    #         # ランダムに占う座標を決める
    #         i = randint(0, N - 1)
    #         j = randint(0, N - 1)
    #         if f"{i} {j}" in fields:
    #             continue
    #         fields.add(f"{i} {j}")
    #         k -= 1
    #     # 占う
    #     times -= 1
    #     command = f"q {len(fields)} {' '.join(fields)}"
    #     print(command)
    #     debug(f"{AHC30} {command}")
    #     # 結果を P に反映する
    #     result = int(input())
    #     oil_value = result / len(fields)
    #     for dx, dy in oil:
    #         x = sx + dx
    #         y = sy + dy
    #         P[x][y] = (P[x][y] * C[x][y] + oil_value) / (C[x][y] + 1)
    #         C[x][y] += 1

    debug_p(to_int=-1)
    debug(f"{AHC30} ---------------------------")
    debug_p(to_int=[0.1, 0.3, 0.4, 10])  # type: ignore

    # 最も高確率で油田がある場所を p 1 で指定し a 回掘る
    a = 5
    q = []
    for i in range(N):
        for j in range(N):
            q.append((-P[i][j], i, j))
    heapq.heapify(q)
    for _ in range(a):
        _, best_i, best_j = heapq.heappop(q)
        print(f"q 1 {best_i} {best_j}")
        debug(f"{AHC30} q 1 {best_i} {best_j}")
        result = int(input())
        D[best_i][best_j] = result

    debug_d()

    # これまでの掘って得られた情報と確率を元に重ね合わせる油田を決める
    # ! ここからする -> やっぱり一旦路線変更
    exit()
    result = int(input())
    D[best_i][best_j] = result
    # 結果で次にどうするか決める
    if result == 0:
        # 油田がなかった
        debug(f"{AHC30} {result=}")
        exit()
    else:
        # 油田があったので周辺を占い確度を高める
        window = N // 5
        times2 = 20
        while times2 > 0:
            # ランダムに配置する油田のインデックスを決める
            i = randint(0, M - 1)
            oil = oils[i]
            height, width = oil_ends[i]
            # 左上の座標をランダムに求める
            a, b = max(0, best_i - window), min(N - 1 - height, best_i + window)
            if a > b:
                times2 -= 1
                continue
            sx = randint(a, b)
            a, b = max(0, best_j - window), min(N - 1 - width, best_j + window)
            if a > b:
                times2 -= 1
                continue
            # debug(f"{a=}, {b=}, {best_j=}, {width=}, {window=}")
            sy = randint(a, b)
            # 占う
            times2 -= 1
            fields = []
            for dx, dy in oil:
                x = sx + dx
                y = sy + dy
                # debug(f"{x=}, {y=}")
                assert 0 <= x < N and 0 <= y < N
                fields.append(f"{x} {y}")
            command = f"q {len(fields)} {' '.join(fields)}"
            print(command)
            debug(f"{AHC30} {command}")
            # 結果を P に反映する
            result = int(input())
            oil_value = result / len(fields)
            for dx, dy in oil:
                x = sx + dx
                y = sy + dy
                P[x][y] = (P[x][y] * C[x][y] + oil_value) / (C[x][y] + 1)
                C[x][y] += 1
        # 掘った箇所から各油田の形と推定しスコアを得る
        best_score, best_oil_idx, best_inner_idx = -1.0, -1, -1
        for i in range(M):
            score, inner_idx = estimate(best_i, best_j, i)
            if best_score < score:
                best_score = score
                best_oil_idx = i
                best_inner_idx = inner_idx
        debug(f"{AHC30} {best_score=}, {best_oil_idx=}, {best_inner_idx=}")
        # 想定される油田の形を (best_i, best_j) から近い場所から掘る
        best_oil = oils[best_oil_idx]
        start_idx = best_inner_idx
        q = deque([])
        been = [False] * len(best_oil)
        been[start_idx] = True
        while q:
            at = q.popleft()
            been[at] = True
    debug(f"{AHC30} ---------------------------")
    debug_p(D, to_int=-1)

    # コストを確認する
    print("c")


def main() -> None:
    global N, M, eps, G, S
    line = input().split()
    N = int(line[0])
    M = int(line[1])
    eps = float(line[2])
    G = [[[] for _ in range(N * N + 1)] for _ in range(M)]
    S = [0] * M
    for m in range(M):
        line = input().split()
        S[m] = int(line[0])
        ps = []
        end_x, end_y = -1, -1
        for i in range(int(line[0])):
            ps.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
            end_x = max(end_x, int(line[2 * i + 1]))
            end_y = max(end_y, int(line[2 * i + 2]))
        oils.append(ps)
        oil_ends.append((end_x, end_y))

        T = [[-1] * (end_y + 1) for _ in range(end_x + 1)]
        for i, (x, y) in enumerate(ps):
            T[x][y] = i
        dpprint(T)
        for i in range(end_x + 1):
            for j in range(end_y + 1):
                if T[i][j] == -1:
                    continue
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = i + dx, j + dy
                    if 0 <= nx <= end_x and 0 <= ny <= end_y and T[nx][ny] != -1:
                        G[m][T[i][j]].append(T[nx][ny])
    # debug(f"{AHC30} {oil_ends=}")  # ! comment when submittion

    # play_random()
    # play_all_dig()
    if (N**2)**M < 2500000:
        play_brute_force()
    else:
        # play_all_dig()
        play_all_dig_with_counting()


if __name__ == "__main__":
    main()
