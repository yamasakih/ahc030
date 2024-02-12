import sys
from typing import Final
from random import randint
from pprint import pprint
from functools import partial


def debug(*args, end="\n") -> None:  # type: ignore
    print(*args, end=end, file=sys.stderr)


dpprint = partial(pprint, stream=sys.stderr)


AHC30: Final[str] = "\033[32m[AHC30]\033[0m"

# prior information
N: int
M: int
eps: float
oils: list[list[tuple[int, int]]] = []
P: list[list[float]]
C: list[list[float]]


def debug_p(to_int: bool = True) -> None:
    if to_int:
        for p in P:
            debug(f"{AHC30} {' '.join('0' if pi < 1 else str(int(pi)) for pi in p)}")
    else:
        for p in P:
            # debug(f"{AHC30} {' '.join(str(pi) for pi in p)}")
            debug(f"{AHC30} {' '.join(format(pi, '.2f') for pi in p)}")


def play_random() -> None:
    global C, P
    P = [[0] * N for _ in range(N)]
    C = [[0] * N for _ in range(N)]
    # ランダムな場所、ランダムな大きさの矩形で何回か占う
    times = 10000
    while times > 0:
        # 左上、右下の座標をランダムに求める
        left_up_x = randint(0, N - 1)
        right_bottom_x = randint(0, N - 1)
        if left_up_x == right_bottom_x:
            continue
        if left_up_x > right_bottom_x:
            left_up_x, right_bottom_x = right_bottom_x, left_up_x
        left_up_y = randint(0, N - 1)
        right_bottom_y = randint(0, N - 1)
        if left_up_y == right_bottom_y:
            continue
        if left_up_y > right_bottom_y:
            left_up_y, right_bottom_y = right_bottom_y, left_up_y
        # 占う
        times -= 1
        fields = []
        for x in range(left_up_x, right_bottom_x + 1):
            for y in range(left_up_y, right_bottom_y + 1):
                fields.append(f"{x} {y}")
        command = f"q {len(fields)} {' '.join(fields)}"
        print(command)
        debug(f"{AHC30} {command}")
        result = int(input())
        # 結果を P に反映する
        oil_value = result / len(fields)
        for x in range(left_up_x, right_bottom_x + 1):
            for y in range(left_up_y, right_bottom_y + 1):
                P[x][y] = (P[x][y] * C[x][y] + oil_value) / (C[x][y] + 1)
                C[x][y] += 1
    debug_p(to_int=False)
    debug(f"{AHC30} ---------------------------")
    debug_p(to_int=True)
    # コストを確認する
    print("c")


def main() -> None:
    global N, M, eps
    line = input().split()
    N = int(line[0])
    M = int(line[1])
    eps = float(line[2])
    for _ in range(M):
        line = input().split()
        ps = []
        for i in range(int(line[0])):
            ps.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
        oils.append(ps)

    play_random()


if __name__ == "__main__":
    main()
