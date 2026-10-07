# -*- coding: utf-8 -*-
"""配对统计：两条臂在**同一批样本**上的差值，带区间与符号检验。

用法：
    python paired_stats.py --a arm_a.jsonl --b arm_b.jsonl [--key mean_logp] [--ids id]
    python paired_stats.py --a arm_a.jsonl --b arm_b.jsonl --subset ids.txt   # 只在这批 id 上比

输入：两份 jsonl，**每行一个样本**，至少要有 id 字段 + 一个数值字段（默认 `mean_logp`）。
输出：配对差值 d̄ / sd / SE / CI95 + 符号检验 p + **每一侧的绝对档分布**（若字段是 0/1/2 的档位）。

⚠️ 三条口径：
  1. **按 id 配对**（不是按行号）—— 顺序不同也要能对上；缺一边的行自动剔除并报数。
  2. **绝对档分布要和 d̄ 一起看**：一侧贴地板/天花板时，d̄ 会被压住（饱和），只看 d̄ 会误判。
  3. **d̄ 的方向**：正 = A 这一侧更大（「更好」与否由你的事先定义决定 —— 先定义，再看数）。
"""
import argparse
import io
import json
import math
import sys
from math import comb

# ⚠️ Windows 控制台的 stdout 默认是 GBK ⇒ 中文 + 数学符号（− ❗ →）会把脚本打崩。
#    这一行是实测踩出来的：本文件第一版在 Windows 上直接 UnicodeEncodeError 退出。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def read(path, key, idkey):
    """返回 (按 id 的取值, 读入行数, 缺字段的行数) —— 后两个用来区分
    「文件是空的」与「字段名写错了」，这两种都会让结果看起来像「没有数据」。"""
    out, nlines, nbad = {}, 0, 0
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        nlines += 1
        r = json.loads(line)
        if idkey not in r or key not in r:
            nbad += 1
            continue
        out[r[idkey]] = float(r[key])
    return out, nlines, nbad


def sign_test(d):
    """二项精确双侧检验：只数正负号。"""
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return pos, neg, min(p, 1.0)


def dist(vals):
    """如果这些值是 0/1/2 这种小整数，报绝对档分布 —— 饱和与否一眼看得出来。"""
    if not vals:
        return None
    if all(abs(v - round(v)) < 1e-9 and 0 <= v <= 9 for v in vals):
        c = {}
        for v in vals:
            c[int(v)] = c.get(int(v), 0) + 1
        return "{" + ", ".join("%d:%d" % (k, c[k]) for k in sorted(c)) + "}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--key", default="mean_logp")
    ap.add_argument("--ids", default="id")
    ap.add_argument("--subset", default=None, help="只在这份文件的 id 列表上比（每行一个 id）")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    a = ap.parse_args()

    # 先查路径 —— 否则下面报的是 FileNotFoundError 的栈，而不是「哪一条不对」
    import os
    for p in [a.a, a.b] + ([a.subset] if a.subset else []):
        if not os.path.exists(p):
            print("[!] 找不到文件：%s" % p)
            return 1

    A, la, ra = read(a.a, a.key, a.ids)
    B, lb, rb = read(a.b, a.key, a.ids)
    keys = sorted(set(A) & set(B))
    if a.subset:
        want = [l.strip() for l in io.open(a.subset, encoding="utf-8") if l.strip()]
        keys = [k for k in keys if k in set(want)]
    if len(keys) < 2:
        print("[!] 配对后不足两条 —— 分三种情况查：")
        print("    A: 读入 %d 行，其中 %d 行缺 `%s` 或 `%s`" % (la, ra, a.ids, a.key))
        print("    B: 读入 %d 行，其中 %d 行缺 `%s` 或 `%s`" % (lb, rb, a.ids, a.key))
        print("    配对上了 %d 条；A 独有 %d，B 独有 %d"
              % (len(keys), len(set(A) - set(B)), len(set(B) - set(A))))
        print("    ⇒ 读入 0 行 = 路径不对；读入 >0 但全缺字段 = 字段名不对；都有但配不上 = id 不是同一套")
        return 1

    d = [A[k] - B[k] for k in keys]
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    lo, hi = m - 1.96 * se, m + 1.96 * se
    pos, neg, p = sign_test(d)
    zeros = n - pos - neg

    print("=== 配对差值（%s - %s）===" % (a.label_a, a.label_b))
    print("字段 = %s | 按 %s 配对 | n = %d%s"
          % (a.key, a.ids, n,
             "" if set(A) == set(B) else "（有一条边单边持有、已剔除：A 独有 %d · B 独有 %d）"
             % (len(set(A) - set(B)), len(set(B) - set(A)))))
    print("d̄ = %+.4f | sd = %.4f | SE = %.4f" % (m, sd, se))
    print("CI95 = [%+.4f, %+.4f]   %s" % (lo, hi, "← 不含 0" if lo * hi > 0 else "← 含 0"))
    print("符号：d>0 %d 行 · d=0 %d 行 · d<0 %d 行 | 二项精确双侧 p = %.4f" % (pos, zeros, neg, p))
    da, db = dist([A[k] for k in keys]), dist([B[k] for k in keys])
    if da:
        print("每一侧的绝对档分布：%s %s · %s %s" % (a.label_a, da, a.label_b, db))
        print("  ⚠️ 若某一侧大量落在顶档或底档，d̄ 会被压住（饱和）—— 这时要读档分布，不要只读 d̄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
