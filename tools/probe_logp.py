# -*- coding: utf-8 -*-
"""教师强制探针：量「模型自己会不会写出这条现成答案」。

为什么单独有这一件：生成式评测（让模型写、再让人或模型判）只能答「写得好不好」；
而「它到底有没有把这条答案学进去」是另一个问题，而且**不用人打分**就能答 ——
把现成答案喂回去，看它给这些 token 多大概率。**0 训练、几秒钟、每行方向一致。**

用法：
    python probe_logp.py --model <HF 模型名> [--adapter <LoRA 目录|base>] \\
        --data <jsonl> [--field chosen] --out <jsonl>

数据行形状：至少要有 `situation`（题面）与 `chosen`（要打分的目标文本）；
也可用 `--field rejected` 去量「差的那一侧」。

⚠️ 三条口径（别踩）：
  1. **长度要归一化**：默认报 `mean_logp = 总对数概率 / token 数`。不归一的话长答案天生吃亏。
  2. **绝对值不可跨模型比**：4bit 量化下只有「同一量化口径下的臂间差」可读。
  3. **prompt 必须与你的生成探针逐字同源**，否则量的是另一个分布。
"""
import argparse
import io
import json
import sys

# ⚠️ Windows 控制台 stdout 默认 GBK ⇒ 中文 + 数学符号（− ❗ →）会把脚本打崩
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


INSTR = ("下面是一段情境。请对它下判断，严格按两行写：\n"
         "第一行的词头是「读数：」，后面写这条情境里的事实。\n"
         "第二行的词头是「判据：」，后面写你从这些事实得出的判断、以及紧接着该做的动作。\n"
         "上面是格式说明，不是要你照抄的句子 —— 回答里只保留「读数：」「判据：」两个词头，\n"
         "其余每个字都要跟这段情境有关，不要重复我这里的说明文字。\n"
         "不要问澄清问题，不要给通用建议，不要写代码。\n\n"
         "情境：")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型名或本地目录（基座）")
    ap.add_argument("--adapter", default="base", help="LoRA 目录；传 base = 不挂适配器")
    ap.add_argument("--data", required=True)
    ap.add_argument("--field", default="chosen", help="要打分的目标字段（chosen / rejected / …）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instr", default=None, help="覆盖默认指令（要与生成探针逐字一致）")
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    instr = a.instr if a.instr is not None else INSTR
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                                 device_map="auto", trust_remote_code=True)
    if a.adapter not in ("base", "", None):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    recs = [json.loads(l) for l in io.open(a.data, encoding="utf-8") if l.strip()]
    print("模型 = %s | 适配器 = %s | 字段 = %s | 条数 = %d"
          % (a.model, a.adapter, a.field, len(recs)))

    rows, bad = [], 0
    for i, r in enumerate(recs):
        tgt_text = r.get(a.field)
        if not tgt_text:
            bad += 1
            continue
        prompt = tok.apply_chat_template([{"role": "user", "content": instr + r.get("situation", "")}],
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(prompt + tgt_text, add_special_tokens=False)["input_ids"]
        # 目标 = full 去掉 prompt 前缀；边界处可能发生 token 合并 ⇒ 试 ±1
        cut = None
        for d in (0, -1, 1):
            L = len(p_ids) + d
            if 0 < L < len(f_ids) and f_ids[:L] == p_ids[:L]:
                cut = L
                break
        if cut is None or len(f_ids) > a.max_len:
            bad += 1
            print("  [%2d] %s 切不开或超长，跳过" % (i + 1, r.get("id")))
            continue
        tgt = f_ids[cut:]
        with torch.no_grad():
            logits = model(torch.tensor([f_ids]).to(model.device)).logits[0]
        seg = logits[cut - 1:len(f_ids) - 1, :].float()      # logits[t] 预测 token t+1
        lp = torch.log_softmax(seg, dim=-1)
        per = lp.gather(1, torch.tensor(tgt).to(model.device).unsqueeze(1)).squeeze(1)
        s, n = float(per.sum()), int(len(tgt))
        rows.append({"id": r.get("id"), "n_tok": n, "n_char": len(tgt_text),
                     "sum_logp": s, "mean_logp": s / n})
        if (i + 1) % 10 == 0:
            print("  [%2d/%d] ..." % (i + 1, len(recs)))

    with io.open(a.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rows)
    if not n:
        print("[!] 一条都没算出来，检查字段名与数据")
        return 1
    ml = sorted(r["mean_logp"] for r in rows)
    print("---- 汇总（%s · 字段 %s）----" % (a.adapter, a.field))
    print("  条数 = %d | 跳过 = %d | 平均 tok = %d"
          % (n, bad, sum(r["n_tok"] for r in rows) // n))
    print("  mean_logp 均值 = %.4f | 中位 = %.4f | 最小 = %.4f | 最大 = %.4f"
          % (sum(ml) / n, ml[n // 2], ml[0], ml[-1]))
    print("[OK] -> %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
