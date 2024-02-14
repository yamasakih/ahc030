import sys
from pathlib import Path
import pandas as pd

def debug(*args, end="\n"):
    print(*args, end=end, file=sys.stderr)


dir = Path("tests/in")
files = list(dir.glob("*.txt"))
debug(f"{files=}")

info = {}
# N, M, EPS, 各油田の大きさ, 油田の大きさの min, median, max, 油田の重なり具合を取得
for file in files:
    with open(file) as f:
        N_, M_, EPS_ = f.readline().split()
        N, M, EPS = int(N_), int(M_), float(EPS_)
        S = []
        for _ in range(M):
            line = f.readline().split()
            # debug(f"{line=}")
            S.append(int(line[0]))
        S = sorted(S)
        min_S = S[0]
        med_S = S[M//2]
        max_S = S[-1]
        # 総油田量 / 面積 = 重なり具合と適宜して計算
        for _ in range(M):
            f.readline()
        sum_oils = sum(S)
        s = 0
        for _ in range(N):
            s += sum([int(i > 0) for i in map(int, f.readline().split())])
        overlap = sum_oils / s
        # debug(f"{sum_oils=}, {s=}, {overlap=}")
        # debug(f"{file.stem=}, {N=}, {M=}, {EPS=}, {min_S=}, {med_S=}, {max_S=}, {overlap=}")
        # # 各情報を dict で保存
        info[file.stem] = dict(N=N, M=M, EPS=EPS, min_S=min_S, med_S=med_S, max_S=max_S, overlap=overlap)

# debug(f"{info=}")

# pandas で dataframe にする
df = pd.DataFrame(info).T
# N, M, min_S, med_S, max_S は int なので int に変換
df = df.astype({"N": int, "M": int, "min_S": int, "med_S": int, "max_S": int, "overlap": float})
# overlap を小数第２位までに丸める
df["overlap"] = df["overlap"].round(2)

# 各 Trial のスコアをカラムとして追加する
dir = Path("tests/out")
files = list(dir.glob("*/summary.log"))
for file in files:
    name = file.parent.name.replace("trial", "tri")
    df[name] = [float(f) for f in file.read_text().strip().split("\n")]
    df[name] = df[name].round(2)
debug(f"{df}")

df.to_csv("tests/info.tsv", sep="\t")
