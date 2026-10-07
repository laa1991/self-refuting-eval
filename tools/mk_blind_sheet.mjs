// 造一张盲表：把两条臂的输出随机分到 A / B，并产出 key.json（判读者不许读它）。
//
// 用法：
//   node mk_blind_sheet.mjs --a arm_a.jsonl --b arm_b.jsonl --out _blind_r01 \
//        [--seed 20261003] [--situation situation] [--text arm_out] [--ref chosen] [--n 0]
//
// 输入两份 jsonl，**按 id 配对**，每行至少要有：情境字段、本臂的输出文本字段。
// 可选 `--ref <字段>`：把「人类当年写的那一条」放进表里当参考线 ——
//   ⚠️ 只有当两条臂都不是参考线本身时才加它（否则是同义反复：判读者一眼对上）。
//
// 产物：
//   <out>/sheet.md       给判读者看的表（不含任何臂名信息）
//   <out>/judge_brief.md 任务书（由 judge_brief.template.md 生成，{{…}} 替换）
//   <keys>/key.json      每行 A/B 分别是哪条臂 —— **判读者不许读**
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const arg = (k, d) => {
  const i = process.argv.indexOf('--' + k);
  return i > -1 ? process.argv[i + 1] : d;
};
const A_F = arg('a'), B_F = arg('b'), OUT = arg('out'), SEED = Number(arg('seed', 20261003));
const F_SIT = arg('situation', 'situation'), F_TXT = arg('text', 'arm_out'), F_REF = arg('ref', null);
// 臂名要写进 key —— 下游分析器靠它认哪一侧是哪条臂。默认取文件名（去掉扩展名）。
const NAME_A = arg('name-a', path.basename(A_F).replace(/\.jsonl?$/, ''));
const NAME_B = arg('name-b', path.basename(B_F).replace(/\.jsonl?$/, ''));
const N = Number(arg('n', 0));
if (!A_F || !B_F || !OUT) { console.error('用法：node mk_blind_sheet.mjs --a <a.jsonl> --b <b.jsonl> --out <目录>'); process.exit(2); }

const read = p => fs.readFileSync(p, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
const A = read(A_F), B = read(B_F);
const BbyId = new Map(B.map(r => [r.id, r]));
let rows = A.filter(r => BbyId.has(r.id)).map(r => ({ a: r, b: BbyId.get(r.id) }));
if (N > 0) rows = rows.slice(0, N);
if (rows.length < 2) { console.error(`[!] 配对后只有 ${rows.length} 行（A ${A.length} / B ${B.length}）—— 检查 id 是否同一套`); process.exit(1); }

// 固定种子的洗牌（可复算：同一个种子 + 同一批数据 ⇒ 同一张表）
const mulberry32 = a => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const rng = mulberry32(SEED);

const key = {}; const body = []; let nA = 0;
for (const { a, b } of rows) {
  const flip = rng() < 0.5;
  key[a.id] = { A: flip ? NAME_B : NAME_A, B: flip ? NAME_A : NAME_B };
  if (flip) nA++;
  const sit = a[F_SIT] ?? '';
  const ta = a[F_TXT] ?? '', tb = b[F_TXT] ?? '';
  const ref = F_REF ? (a[F_REF] ?? b[F_REF]) : null;
  body.push(`### ${body.length + 1}. \`${a.id}\`\n\n**情境**：${sit}\n\n`
    + (ref ? `**人类当年写的那一条（参考，不打分）**：${ref}\n\n` : '')
    + `**A**：${flip ? tb : ta}\n\n**B**：${flip ? ta : tb}\n\n两侧各打一个分（0 / 1 / 2）：A = ____  B = ____\n`);
}
// ⚠️ 用 fileURLToPath 而不是 new URL(...).pathname —— 后者会把中文/空格百分号编码，
//    而这份包的路径里就带中文目录名（实测会 readFileSync 失败）。
const HERE = path.dirname(fileURLToPath(import.meta.url));
const head = fs.readFileSync(path.join(HERE, 'judge_brief.template.md'), 'utf8');
const dirName = path.basename(OUT), keyName = '_keys_' + dirName.replace(/^_blind_/, '');
fs.mkdirSync(OUT, { recursive: true });
fs.mkdirSync(keyName, { recursive: true });
fs.writeFileSync(path.join(OUT, 'sheet.md'), body.join('\n---\n\n') + '\n', 'utf8');
fs.writeFileSync(path.join(keyName, 'key.json'), JSON.stringify(key, null, 1), 'utf8');
fs.writeFileSync(path.join(OUT, 'judge_brief.md'),
  head.replaceAll('{{KEY_DIR}}', keyName).replaceAll('{{SHEET_DIR}}', dirName).replaceAll('{{N}}', String(rows.length)).replace('{{HEAD}}', ''),
  'utf8');
const med = a => { const s = [...a].sort((x, y) => x - y); return s[s.length >> 1]; };
console.log(`块 = ${rows.length} · key = ${Object.keys(key).length} · 两侧平衡：A 侧 ${nA} / ${rows.length - nA}`);
console.log(`臂名（写进 key 的）：A 文件 = ${NAME_A} · B 文件 = ${NAME_B}`);
console.log(`长度中位：臂A(文件 ${path.basename(A_F)}) ${med(A.map(r => (r[F_TXT] || '').length))} · 臂B(${path.basename(B_F)}) ${med(B.map(r => (r[F_TXT] || '').length))}  ← ⚠️ 差得多就先看任务书里的「长度不计分」有没有写`);
console.log(`产物：${OUT}/sheet.md + ${OUT}/judge_brief.md · key 在 ${keyName}/key.json（判读者不许读）`);
