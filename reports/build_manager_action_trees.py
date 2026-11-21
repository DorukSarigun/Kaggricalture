#!/usr/bin/env python3
"""Build a single offline HTML of manager action trees from a training run."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "feudal_manager_issue_gold_dag_v7_01"
OUT = Path(__file__).with_name("manager_action_trees.html")
MOVES = frozenset({"NORTH", "SOUTH", "EAST", "WEST"})
CARE = frozenset({"WATER", "FEED", "CARE"})


def load_json(path):
    return json.loads(path.read_text())


def episode_dir(n):
    return RUN / "episodes" / f"episode_{n:06d}"


def op_name(action):
    if not action:
        return "—"
    return " ".join(str(x) for x in action[:2])


def index_decisions(rows, *, skip_boundary=False):
    out = {}
    if isinstance(rows, dict):
        rows = rows.get("decisions") or list(rows.values())
    for row in rows:
        if not isinstance(row, dict):
            continue
        if skip_boundary and (row.get("kind") == "boundary" or row.get("boundary")):
            continue
        out[row["decision_id"]] = row
    return out


def lookup(did, manager, crew):
    if did is None:
        return "book", "geçiş", None
    if did % 2 == 0:
        row = manager.get(did // 2)
        action = (row or {}).get("action") or []
        return "manager", op_name(action) or f"dec {did // 2}", did // 2
    row = crew.get(did // 2)
    action = (row or {}).get("action") or []
    return "crew", op_name(action) or f"crew {did // 2}", did // 2


def gold_for(did, table):
    if did is None:
        return 0.0
    if did in table:
        return table[did]
    if did % 2 == 0:
        return table.get(did // 2, 0.0)
    return 0.0


def walk(nodes, nid, manager, crew, table, ancestors=()):
    if nid is None or nid in ancestors:
        return None
    node = nodes[nid]
    role, name, orig = lookup(node.get("decision_id"), manager, crew)
    kids = []
    nxt = ancestors + (nid,)
    for parent, _weight in node.get("parents") or ():
        child = walk(nodes, parent, manager, crew, table, nxt)
        if child:
            kids.append(child)
    return {
        "role": role,
        "name": name,
        "gold": round(float(gold_for(node.get("decision_id"), table)), 4),
        "id": orig,
        "kids": kids,
    }


def collapse(tree):
    if not tree:
        return tree
    kids = [collapse(k) for k in tree["kids"]]
    name = tree["name"]
    head = name.split()[0] if name else ""
    if tree["role"] == "crew" and head in MOVES | CARE and len(kids) == 1:
        count = 1
        child = kids[0]
        while (
            child["role"] == "crew"
            and (child["name"].split() or [""])[0] == head
            and len(child["kids"]) == 1
            and abs(child["gold"]) < 1e-9
        ):
            count += 1
            child = child["kids"][0]
        label = f"{head} ×{count}" if count > 1 else head
        return {**tree, "name": label, "kids": [child]}
    return {**tree, "kids": kids}


def signature(tree):
    if not tree:
        return ""
    return f"{tree['role'][0]}:{tree['name']}[{','.join(signature(k) for k in tree['kids'])}]"


def origin_of(alloc):
    origins = alloc.get("origins") or ()
    text = " ".join(str(x) for x in origins)
    if any(str(x).startswith(("plant:", "animal:")) for x in origins):
        return "üretilmiş"
    if "purchase" in text or "seed" in text or "animal_stock" in text:
        return "alış"
    if not origins:
        return "doğrudan"
    return "diğer"


def kind_tr(kind):
    return {
        "sale": "satış",
        "unsold_inventory": "satılmayan stok",
        "unrealized_input": "hasatsız girdi",
        "direct_cost": "HIRE/arazi",
    }.get(kind, kind)


def extract_episode(n):
    folder = episode_dir(n)
    graph = load_json(folder / "issue_production_graph.json")
    credit = load_json(folder / "issue_sale_credit.json")
    summary = load_json(folder / "summary.json")
    manager = index_decisions(load_json(folder / "manager_decisions.json"), skip_boundary=True)
    crew = index_decisions(load_json(folder / "crew_decisions.json"))
    issues = load_json(folder / "issues.json")
    seq = issues.values() if isinstance(issues, dict) else issues
    items, status = Counter(), Counter()
    for issue in seq:
        if isinstance(issue, dict):
            items[issue.get("item")] += 1
            status[issue.get("status")] += 1
    ops = Counter()
    for row in manager.values():
        action = row.get("action") or ["?"]
        ops[action[0]] += 1
    nodes = graph["action_nodes"]
    grouped = {}
    for alloc in credit["allocations"]:
        table = defaultdict(float)
        for share in alloc.get("shares") or ():
            table[share["decision_id"]] += share["gold"]
        tree = collapse(walk(nodes, alloc.get("action_root"), manager, crew, table))
        sig = signature(tree)
        bucket = grouped.setdefault(
            sig,
            {
                "sig": sig,
                "count": 0,
                "gold": 0.0,
                "kind": Counter(),
                "origin": Counter(),
                "example": tree,
                "has_issue": "ISSUE" in sig,
                "has_crew": ":PLANT" in sig or ":PLACE" in sig or ":HARVEST" in sig,
            },
        )
        bucket["count"] += 1
        bucket["gold"] += alloc.get("net_profit") or 0.0
        bucket["kind"][kind_tr(alloc.get("kind"))] += 1
        bucket["origin"][origin_of(alloc)] += 1
    groups = []
    for bucket in grouped.values():
        groups.append(
            {
                "sig": bucket["sig"],
                "count": bucket["count"],
                "gold": round(bucket["gold"], 2),
                "kind": dict(bucket["kind"]),
                "origin": dict(bucket["origin"]),
                "example": bucket["example"],
                "has_issue": bucket["has_issue"],
                "has_crew": bucket["has_crew"],
            }
        )
    groups.sort(key=lambda g: (not g["has_issue"] and not g["has_crew"], -g["count"], -abs(g["gold"])))
    issue_sinks = sum(g["count"] for g in groups if g["has_issue"])
    flip_sinks = sum(g["count"] for g in groups if (not g["has_issue"] and "SELL" in g["sig"] and "BUY" in g["sig"]))
    return {
        "episode": n,
        "money": summary["money"],
        "rival": summary["opponent_money"],
        "net": credit["net_cash"],
        "sales": summary["sale_count"],
        "issues": summary["issue_count"],
        "fulfilled": summary["fulfilled_units"],
        "unmet": summary["unmet_issue_units"],
        "throughput": summary["throughput_credit_sum"],
        "status": dict(status),
        "items": dict(items),
        "ops": dict(ops),
        "alloc": len(credit["allocations"]),
        "unique": len(groups),
        "issue_sinks": issue_sinks,
        "flip_sinks": flip_sinks,
        "groups": groups,
    }


def training_overview():
    episodes, updates = [], []
    for line in (RUN / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "episode":
            episodes.append(row)
        elif row.get("event") == "update":
            updates.append(row)

    def window(a, b):
        xs = episodes[a:b]
        n = len(xs) or 1
        return {
            "span": f"{xs[0]['episode']}–{xs[-1]['episode']}",
            "money": round(sum(x["money"] for x in xs) / n, 1),
            "net": round(sum(x["economic_net_cash"] for x in xs) / n, 1),
            "sales": round(sum(x["sale_count"] for x in xs) / n, 1),
            "issues": round(sum(x["issue_count"] for x in xs) / n, 1),
            "thru": round(sum(x["throughput_credit_sum"] for x in xs) / n, 1),
            "zero": sum(x["money"] == 0 for x in xs),
        }

    return {
        "run": str(RUN.relative_to(ROOT)),
        "version": load_json(RUN / "config.json").get("version"),
        "optimizer": load_json(RUN / "config.json").get("optimizer"),
        "n": len(episodes),
        "updates": len(updates),
        "money": [x["money"] for x in episodes],
        "net": [x["economic_net_cash"] for x in episodes],
        "sales": [x["sale_count"] for x in episodes],
        "thru": [x["throughput_credit_sum"] for x in episodes],
        "windows": [window(0, 20), window(180, 200), window(380, 400)],
        "zero": sum(x["money"] == 0 for x in episodes),
        "max_money": max(x["money"] for x in episodes),
        "last": {
            "episode": episodes[-1]["episode"],
            "money": episodes[-1]["money"],
            "rival": episodes[-1]["opponent_money"],
            "net": episodes[-1]["economic_net_cash"],
            "sales": episodes[-1]["sale_count"],
            "thru": episodes[-1]["throughput_credit_sum"],
        },
        "ppo": {
            "first_loss": updates[0]["loss"],
            "last_loss": updates[-1]["loss"],
            "first_term_v": updates[0]["terminal_value_loss"],
            "last_term_v": updates[-1]["terminal_value_loss"],
            "first_chain_u": updates[0]["chain_adv_unit"],
            "last_chain_u": updates[-1]["chain_adv_unit"],
            "last_term_u": updates[-1]["terminal_adv_unit"],
            "last_kl": updates[-1]["kl"],
            "last_grad": updates[-1]["gradient_norm_preclip"],
        },
    }


def html_page(data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aksiyon ağaçları · gold-dag v7</title>
<style>
:root{{--bg:#f5f4ed;--paper:#fffef9;--ink:#20342b;--muted:#57695e;--line:#d9dfd4;--green:#28684d;--mint:#d9ead6;--ochre:#976520;--amber:#fbecd1;--red:#a44733;--dark:#142e25;--mono:ui-monospace,"DejaVu Sans Mono",Consolas,monospace}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}}
h1,h2{{font-family:Georgia,"DejaVu Serif",serif;font-weight:400;letter-spacing:-.04em;line-height:1.15}}
h1{{font-size:clamp(32px,4vw,54px);margin:0 0 12px}}h2{{font-size:22px;margin:0 0 12px}}
.wrap{{max-width:1240px;margin:0 auto;padding:28px 22px 60px}}
.eyebrow{{font:11px/1.4 var(--mono);letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:10px}}
.hero{{background:var(--dark);color:#f2f5e8;padding:32px;border-radius:14px;margin-bottom:22px}}
.hero p{{color:#c7d5c7;max-width:720px}}
.kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:22px}}
.kpi{{background:#203d2d;border:1px solid #466247;border-radius:10px;padding:12px 14px}}
.kpi b{{display:block;font:22px/1.2 var(--mono);margin-top:4px}}
.kpi span{{font-size:11px;color:#c7d5c7}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}}
.muted{{color:var(--muted)}}
.row{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
button,select{{font:inherit;border:1px solid var(--line);background:var(--paper);border-radius:8px;padding:8px 12px;cursor:pointer}}
button[aria-pressed="true"],button.active{{background:var(--green);color:#fff;border-color:var(--green)}}
input[type=search]{{flex:1;min-width:180px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;background:var(--paper)}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);padding:8px;border-bottom:1px solid var(--line)}}
td{{padding:8px;border-top:1px solid var(--line);vertical-align:top}}
.pos{{color:var(--green)}} .neg{{color:var(--red)}}
svg.spark{{width:100%;height:64px;display:block}}
.tree{{font:13px/1.35 var(--mono);margin:8px 0 0}}
.tree .kids{{margin:0;padding:0 0 0 16px;list-style:none;border-left:1px solid #c5d0c2}}
.tree>li{{list-style:none}}
.pill{{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:2px 8px;border:1px solid var(--line);font-size:12px}}
.pill.manager{{background:var(--mint);border-color:#b8d2ba}}
.pill.crew{{background:var(--amber);border-color:#e0c690}}
.pill.book{{background:#edf0e6}}
.gold{{font-variant-numeric:tabular-nums;margin-left:8px}}
.group{{border-top:1px solid var(--line);padding:12px 0}}
.group header{{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}}
.count{{font:12px var(--mono);color:var(--muted)}}
.note{{font-size:13px;color:var(--muted);border-left:3px solid var(--green);padding:8px 12px;background:#eaf0e4;margin:12px 0}}
@media(max-width:900px){{.kpis{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <div class="eyebrow" id="run-label"></div>
    <h1>Net kârın izlediği aksiyon ağaçları</h1>
    <p>Her yaprak bir sink: satış, satılmayan stok veya hasatsız girdi. Altın, o lot’un aksiyon DAG’ındaki manager atalarına gider. Crew düğümü ödenmez; yürüme/sulama zincirleri sıkıştırıldı. Aynı şekil bir kez çizilir, kaç kere tekrarlandığı yanında durur.</p>
    <div class="kpis" id="kpis"></div>
  </header>
  <section class="card">
    <div class="eyebrow">400 maç</div>
    <h2>Kasa toparlandı; terminal hâlâ rakibi kaybediyor</h2>
    <p class="muted" id="story"></p>
    <svg class="spark" id="spark" viewBox="0 0 400 64" preserveAspectRatio="none"></svg>
    <p class="muted" style="font:11px var(--mono);margin:8px 0 0">yeşil: bitiş bakiyesi · kahverengi: net kasa (başlangıç 3000’e göre)</p>
    <div class="table-wrap" style="margin-top:14px"><table id="windows"></table></div>
  </section>
  <section class="card">
    <div class="row">
      <select id="episode"></select>
      <button type="button" data-filter="all" class="active">hepsi</button>
      <button type="button" data-filter="issue">ISSUE yolu</button>
      <button type="button" data-filter="flip">yalnız al-sat</button>
      <button type="button" data-filter="left">satılmayan / hasatsız</button>
      <input type="search" id="q" placeholder="SELL CARROT, BUY_SEED, PLANT…">
    </div>
    <p class="note" id="epnote"></p>
    <div id="groups"></div>
  </section>
</div>
<script id="data" type="application/json">{payload}</script>
<script>
const DATA = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const fmt = (n, d=0) => Number(n).toLocaleString("tr-TR", {{maximumFractionDigits:d, minimumFractionDigits:d}});
const gold = (n) => `<span class="gold ${{n<0?"neg":n>0?"pos":""}}">${{n>0?"+":""}}${{fmt(n,2)}}</span>`;

$("run-label").textContent = DATA.train.run + " · " + DATA.train.version;
$("kpis").innerHTML = [
  ["son kasa", fmt(DATA.train.last.money)],
  ["son net", fmt(DATA.train.last.net)],
  ["son satış", fmt(DATA.train.last.sales)],
  ["0 altın maç", DATA.train.zero + " / " + DATA.train.n],
  ["PPO loss", fmt(DATA.train.ppo.last_loss, 2)],
].map(([k,v]) => `<div class="kpi"><span>${{k}}</span><b>${{v}}</b></div>`).join("");

$("story").textContent = `İlk 20 maç ortalama kasa ${{fmt(DATA.train.windows[0].money)}} ve net ${{fmt(DATA.train.windows[0].net)}}. Son 20 maç kasa ${{fmt(DATA.train.windows[2].money)}}, net ${{fmt(DATA.train.windows[2].net)}}, throughput ${{fmt(DATA.train.windows[2].thru,1)}}. Optimizer RMS birimleri: terminal ${{fmt(DATA.train.ppo.last_term_u)}} · zincir ${{fmt(DATA.train.ppo.last_chain_u,2)}}. Kayıtlı kredi hâlâ altın.`;

$("windows").innerHTML = `<thead><tr><th>Aralık</th><th>Kasa</th><th>Net</th><th>Satış</th><th>ISSUE</th><th>Throughput</th><th>0 altın</th></tr></thead><tbody>` +
  DATA.train.windows.map(w => `<tr><td>${{w.span}}</td><td>${{fmt(w.money)}}</td><td class="${{w.net<0?"neg":"pos"}}">${{fmt(w.net)}}</td><td>${{fmt(w.sales)}}</td><td>${{fmt(w.issues)}}</td><td>${{fmt(w.thru,1)}}</td><td>${{w.zero}}</td></tr>`).join("") + "</tbody>";

function spark() {{
  const m = DATA.train.money, n = DATA.train.net;
  const max = Math.max(...m, ...n.map(Math.abs), 1);
  const pts = (xs, color) => xs.map((v,i) => `${{i*400/(xs.length-1)}},${{32 - 28*v/max}}`).join(" ");
  $("spark").innerHTML = `<polyline fill="none" stroke="#7eac84" stroke-width="1.6" points="${{pts(m)}}"/><polyline fill="none" stroke="#976520" stroke-width="1.2" points="${{pts(n)}}"/>`;
}}
spark();

const sel = $("episode");
DATA.episodes.forEach((e,i) => {{
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `maç ${{e.episode}} · kasa ${{fmt(e.money)}} · ${{e.unique}} şekil / ${{e.alloc}} sink`;
  sel.appendChild(o);
}});
sel.value = String(DATA.episodes.length - 1);

function treeHTML(node) {{
  if (!node) return "";
  const kids = (node.kids || []).map(treeHTML).join("");
  return `<li><span class="pill ${{node.role}}">${{node.name}}</span>${{gold(node.gold)}}${{kids ? `<ul class="kids">${{kids}}</ul>` : ""}}</li>`;
}}

let filter = "all";
function matches(g) {{
  const q = $("q").value.trim().toLowerCase();
  if (q && !g.sig.toLowerCase().includes(q) && !JSON.stringify(g.example).toLowerCase().includes(q)) return false;
  if (filter === "issue") return g.has_issue;
  if (filter === "flip") return !g.has_issue && !g.has_crew && g.sig.includes("SELL") && g.sig.includes("BUY");
  if (filter === "left") return Object.keys(g.kind).some(k => k !== "satış");
  return true;
}}

function render() {{
  const ep = DATA.episodes[+sel.value];
  const shown = ep.groups.filter(matches);
  const items = Object.entries(ep.items).map(([k,v]) => k+": "+v).join(" · ") || "—";
  $("epnote").innerHTML = `Maç ${{ep.episode}}: kasa <b>${{fmt(ep.money)}}</b>, net <b>${{fmt(ep.net)}}</b>, ${{ep.sales}} satış, ${{ep.issues}} ISSUE (${{items}}). ${{ep.alloc}} sink → ${{ep.unique}} şekil (${{ep.issue_sinks}} ISSUE yolu, ${{ep.flip_sinks}} al-sat). Gösterilen ${{shown.length}}.`;
  $("groups").innerHTML = shown.map(g => `
    <article class="group">
      <header>
        <div><span class="count">${{g.count}} sink · ${{Object.entries(g.origin).map(([k,v])=>k+" "+v).join(", ")}} · ${{Object.entries(g.kind).map(([k,v])=>k+" "+v).join(", ")}}</span></div>
        <div>${{gold(g.gold)}}</div>
      </header>
      <ul class="tree">${{treeHTML(g.example)}}</ul>
    </article>`).join("") || `<p class="muted">Bu süzgeçte ağaç yok.</p>`;
}}
sel.addEventListener("change", render);
$("q").addEventListener("input", render);
document.querySelectorAll("button[data-filter]").forEach(btn => btn.addEventListener("click", () => {{
  filter = btn.dataset.filter;
  document.querySelectorAll("button[data-filter]").forEach(b => b.classList.toggle("active", b===btn));
  render();
}}));
render();
</script>
</body>
</html>
"""


def main():
    data = {
        "train": training_overview(),
        "episodes": [extract_episode(n) for n in (1, 200, 400)],
    }
    OUT.write_text(html_page(data), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
