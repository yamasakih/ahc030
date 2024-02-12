import sys
from typing import Final


def debug(*args, end="\n") -> None:  # type: ignore
    print(*args, end=end, file=sys.stderr)


AHC30: Final[str] = "\033[32m[AHC30]\033[0m"

# prior information
N: int
M: int
eps: float
oils: list[list[tuple[int, int]]] = []


def play_random() -> None:
    pass


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
