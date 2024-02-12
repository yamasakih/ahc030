import sys

AHC30 = "\033[32m[AHC30]\033[0m"


def debug(*args, end="\n") -> None:  # type: ignore
    print(*args, end=end, file=sys.stderr)


# read prior information
line = input().split()
N = int(line[0])
M = int(line[1])
eps = float(line[2])
fields = []
for _ in range(M):
    line = input().split()
    ps = []
    for i in range(int(line[0])):
        ps.append((int(line[2 * i + 1]), int(line[2 * i + 2])))
    fields.append(ps)

# drill every square
has_oil = []
for i in range(N):
    for j in range(N):
        print("q 1 {} {}".format(i, j))
        debug("{} q 1 {} {}".format(AHC30, i, j))
        resp = input()
        # print(f"{resp=}")
        if resp != "0":
            has_oil.append((i, j))

print(
    "a {} {}".format(
        len(has_oil), " ".join(map(lambda x: "{} {}".format(x[0], x[1]), has_oil))
    )
)
debug(
    "{} a {} {}".format(
        AHC30,
        len(has_oil),
        " ".join(map(lambda x: "{} {}".format(x[0], x[1]), has_oil)),
    )
)
resp = input()
assert resp == "1"
