import sys
from typing import Any, Final, Union
from random import randint
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
            debug(f"{AHC30} q 1 {i} {j}")
            resp = input()
            if resp != "0":
                has_oil.append((i, j))

    oil_positions = " ".join(map(lambda x: f"{x[0]} {x[1]}", has_oil))
    print(f"a {len(has_oil)} {oil_positions}")
    debug(f"{AHC30} a {len(has_oil)} {oil_positions}")
    judge = int(input())
    debug(f"{AHC30} {judge=}")
    assert judge == 1


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
        debug(f"{AHC30} {command}")
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
    global N, M, eps, G
    line = input().split()
    N = int(line[0])
    M = int(line[1])
    eps = float(line[2])
    G = [[[] for _ in range(N * N + 1)] for _ in range(M)]
    for m in range(M):
        line = input().split()
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
    debug(f"{AHC30} {oil_ends=}")

    # play_random()
    # play_brute_force()
    play_all_dig()


if __name__ == "__main__":
    main()
