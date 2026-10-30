"""Zagon backtesta: vse kombinacije add-onov × vsi pari, 70/30 IS/OOS, zapis v bohak.db in results/.

Primer:
    python engine/run.py --strategy example_ema_cross --pairs US30.cash --tf M5
"""
import argparse
import importlib
import inspect
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from engine import db  # noqa: E402
from engine.backtest import metrics, simulate  # noqa: E402
from strategies.base import Strategy  # noqa: E402

LINE, SURFACE, INK, MUTED, OOS_FILL = "#2a78d6", "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe8"


def load_strategy(name: str) -> Strategy:
    mod = importlib.import_module(f"strategies.{name}")
    classes = [c for _, c in inspect.getmembers(mod, inspect.isclass)
               if issubclass(c, Strategy) and c is not Strategy and c.__module__ == mod.__name__]
    if len(classes) != 1:
        sys.exit(f"[NAPAKA] strategies/{name}.py mora vsebovati natanko en razred, ki deduje od Strategy.")
    return classes[0]()


def load_data(data_dir: Path, symbol: str, tf: str) -> tuple[pd.DataFrame, dict]:
    f = data_dir / f"{symbol}_{tf}.parquet"
    m = data_dir / f"{symbol}_meta.json"
    if not f.exists() or not m.exists():
        sys.exit(f"[NAPAKA] Ni podatkov za {symbol} {tf}. Zaženi: python data/download.py --pairs {symbol} --tf {tf}")
    df = pd.read_parquet(f).sort_values("time").reset_index(drop=True)
    return df, json.loads(m.read_text(encoding="utf-8"))


def combo_label(combo: dict) -> str:
    on = [k for k, v in combo.items() if v]
    return "+".join(on) if on else "base"


def plot(tr: pd.DataFrame, split_time, balance: float, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=110, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    if not tr.empty:
        x = pd.concat([pd.Series([tr["entry_time"].iloc[0]]), tr["exit_time"]], ignore_index=True)
        y = [balance] + list(balance + tr["pnl"].cumsum())
        ax.plot(x, y, color=LINE, linewidth=2, solid_capstyle="round")
        ax.axvspan(split_time, max(x.iloc[-1], split_time), color=OOS_FILL, zorder=0, linewidth=0)
        ax.text(split_time, 1.0, "  out-of-sample", transform=ax.get_xaxis_transform(),
                va="top", ha="left", color=MUTED, fontsize=9)
    ax.axhline(balance, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel("Equity (€)", color=MUTED)
    ax.grid(axis="y", color="#e2e1dd", linewidth=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description="Backtest vseh kombinacij add-onov")
    ap.add_argument("--strategy", required=True, help="ime datoteke v strategies/ brez .py")
    ap.add_argument("--pairs", default=",".join(cfg["default_pairs"]))
    ap.add_argument("--tf", default=cfg["default_timeframe"])
    ap.add_argument("--plots", choices=["all", "none"], default="all")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--db", default=str(ROOT / "bohak.db"))
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    strat = load_strategy(args.strategy)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    names = list(strat.addons)
    combos = [dict(zip(names, bits)) for bits in itertools.product([False, True], repeat=len(names))]
    ratio, balance = cfg["in_sample_ratio"], cfg["account_balance"]

    data = {p: load_data(Path(args.data_dir), p, args.tf) for p in pairs}
    split = {p: str(df["time"].iloc[int(len(df) * ratio)]) for p, (df, _) in data.items()}
    od = str(min(df["time"].iloc[0] for df, _ in data.values()))
    do = str(max(df["time"].iloc[-1] for df, _ in data.values()))

    con = db.connect(Path(args.db))
    sid = db.upsert_strategy(con, args.strategy, strat.description, f"strategies/{args.strategy}.py", names)
    tid = db.create_test(con, sid, args.tf, pairs, od, do, split, cfg)
    out_dir = Path(args.results_dir) / str(tid)
    if args.plots == "all":
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test #{tid}: {args.strategy} | {len(combos)} kombinacij × {len(pairs)} parov | {args.tf} | {od} -> {do}")
    rows = []
    for p, (df, meta) in data.items():
        split_time = pd.Timestamp(split[p])
        for combo in combos:
            sig = strat.generate_signals(df, combo)
            tr = simulate(df, sig, meta, cfg, p)
            label = combo_label(combo)
            graf = None
            if args.plots == "all":
                graf = out_dir / f"{p}_{label}.png"
                plot(tr, split_time, balance, f"{args.strategy} | {p} {args.tf} | {label}", graf)
                graf = str(graf.relative_to(ROOT)) if graf.is_relative_to(ROOT) else str(graf)
            is_tr = tr[tr["entry_time"] < split_time] if not tr.empty else tr
            oos_tr = tr[tr["entry_time"] >= split_time] if not tr.empty else tr
            for vz, part in (("in", is_tr), ("out", oos_tr)):
                m = metrics(part, balance)
                db.insert_result(con, tid, p, combo, vz, m, graf)
                rows.append({"par": p, "kombinacija": label, "vzorec": vz, **m})
        con.commit()
    con.close()

    res = pd.DataFrame(rows)
    piv = res.pivot_table(index=["par", "kombinacija"], columns="vzorec",
                          values=["st_tradov", "profit_factor", "expectancy_r", "net_profit"])
    pd.set_option("display.width", 200)
    print(piv.round(2).to_string())
    print(f"\nKončano. test_id={tid}. Analiza: python engine/report.py --test-id {tid}")


if __name__ == "__main__":
    main()
