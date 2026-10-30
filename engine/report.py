"""Analiza testa: top 5 kombinacij in ocena robustnosti.

Primer:
    python engine/report.py --latest
    python engine/report.py --test-id 3 --json

Kriteriji robustnosti (config.json -> robustness):
  1. profitabilna (PF > 1 na celotnem obdobju) na >= 3 od 5 parov
     (pri manj parih: >= 60 % parov; en sam par je označen kot šibek dokaz)
  2. >= 100 tradov na vsakem paru (IS + OOS)
  3. OOS potrjeno: na večini parov OOS PF > 1 in OOS expectancy > 0,
     povprečna OOS expectancy ne pade za več kot max_oos_degradation proti IS
"""
import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def analyse(con, test_id: int) -> dict:
    test = con.execute("SELECT t.*, s.ime FROM testi t JOIN strategije s ON s.id=t.strategija_id WHERE t.id=?",
                       (test_id,)).fetchone()
    if not test:
        sys.exit(f"[NAPAKA] Test {test_id} ne obstaja.")
    cfg = json.loads(test["config"])
    rb = cfg["robustness"]
    rows = con.execute("SELECT * FROM rezultati WHERE test_id=?", (test_id,)).fetchall()
    pairs = json.loads(test["pari"])

    combos: dict[str, dict] = {}
    for r in rows:
        c = combos.setdefault(r["kombinacija"], {})
        c.setdefault(r["par"], {})[r["vzorec"]] = dict(r)

    n_pairs = len(pairs)
    need_pairs = rb["min_profitable_pairs"] if n_pairs >= rb["of_pairs"] else max(1, math.ceil(0.6 * n_pairs))

    out = []
    for key, per_pair in combos.items():
        prof_pairs, enough_trades, oos_ok_pairs = 0, True, 0
        is_exp, oos_exp, oos_trades, oos_net, total_trades = [], [], 0, 0.0, 0
        detail = {}
        for p, d in per_pair.items():
            i, o = d.get("in"), d.get("out")
            n = i["st_tradov"] + o["st_tradov"]
            net = i["net_profit"] + o["net_profit"]
            total_trades += n
            prof_pairs += net > 0
            enough_trades &= n >= rb["min_trades_per_pair"]
            ok = o["profit_factor"] > 1 and o["expectancy_r"] > 0
            oos_ok_pairs += ok
            is_exp.append(i["expectancy_r"])
            oos_exp.append(o["expectancy_r"])
            oos_trades += o["st_tradov"]
            oos_net += o["net_profit"]
            detail[p] = {
                "tradi": n, "is_pf": round(i["profit_factor"], 2), "oos_pf": round(o["profit_factor"], 2),
                "is_exp_r": round(i["expectancy_r"], 3), "oos_exp_r": round(o["expectancy_r"], 3),
                "oos_win_rate": round(o["win_rate"], 3), "oos_max_dd_pct": round(o["max_dd_pct"], 2),
                "net_profit": round(net, 2), "graf": i["graf_pot"],
            }
        avg_is = sum(is_exp) / len(is_exp)
        avg_oos = sum(oos_exp) / len(oos_exp)
        degr_ok = avg_is <= 0 or avg_oos >= avg_is * (1 - rb["max_oos_degradation"])
        oos_confirmed = oos_ok_pairs >= need_pairs and avg_oos > 0 and degr_ok
        crit = {
            "profitabilni_pari": f"{prof_pairs}/{n_pairs} (potrebno {need_pairs})",
            "dovolj_tradov": enough_trades,
            "oos_potrjeno": oos_confirmed,
        }
        robust = prof_pairs >= need_pairs and enough_trades and oos_confirmed
        warnings = []
        if n_pairs < rb["of_pairs"]:
            warnings.append(f"testirano samo na {n_pairs} paru/ih – kriterij '3 od 5' ni v celoti preverljiv")
        if avg_is > 0 and avg_oos < avg_is * 0.5:
            warnings.append("OOS expectancy je pod polovico IS – možen overfitting")
        if not enough_trades:
            warnings.append(f"premalo tradov (<{rb['min_trades_per_pair']}) na vsaj enem paru")
        out.append({
            "kombinacija": json.loads(key), "robustna": robust, "kriteriji": crit,
            "avg_is_exp_r": round(avg_is, 3), "avg_oos_exp_r": round(avg_oos, 3),
            "oos_tradi": oos_trades, "oos_net_profit": round(oos_net, 2), "tradi_skupaj": total_trades,
            "opozorila": warnings, "pari": detail,
        })

    out.sort(key=lambda x: (x["robustna"], x["avg_oos_exp_r"]), reverse=True)
    return {"test_id": test_id, "strategija": test["ime"], "timeframe": test["timeframe"], "pari": pairs,
            "obdobje": [test["od"], test["do"]], "st_kombinacij": len(out),
            "st_robustnih": sum(x["robustna"] for x in out), "top5": out[:5]}


def print_report(r: dict) -> None:
    print(f"Test #{r['test_id']} | {r['strategija']} | {r['timeframe']} | pari: {', '.join(r['pari'])}")
    print(f"Obdobje {r['obdobje'][0]} -> {r['obdobje'][1]} | kombinacij: {r['st_kombinacij']} | "
          f"robustnih: {r['st_robustnih']}\n")
    for k, c in enumerate(r["top5"], 1):
        on = [a for a, v in c["kombinacija"].items() if v] or ["(brez add-onov)"]
        print(f"{k}. {' + '.join(on)}  ->  {'ROBUSTNA' if c['robustna'] else 'ni robustna'}")
        print(f"   IS exp {c['avg_is_exp_r']:+.3f}R | OOS exp {c['avg_oos_exp_r']:+.3f}R | "
              f"OOS tradov {c['oos_tradi']} | OOS net {c['oos_net_profit']:+.2f} €")
        print(f"   kriteriji: {c['kriteriji']}")
        for p, d in c["pari"].items():
            print(f"     {p:<12} tradov {d['tradi']:>5} | PF IS {d['is_pf']:>6} / OOS {d['oos_pf']:>6} | "
                  f"exp OOS {d['oos_exp_r']:+.3f}R | DD OOS {d['oos_max_dd_pct']}% | net {d['net_profit']:+.2f} €")
        for w in c["opozorila"]:
            print(f"   ! {w}")
        print()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Top 5 kombinacij in ocena robustnosti")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test-id", type=int)
    g.add_argument("--latest", action="store_true")
    ap.add_argument("--json", action="store_true", help="izpis v JSON (za agenta)")
    ap.add_argument("--db", default=str(ROOT / "bohak.db"))
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    tid = args.test_id or (con.execute("SELECT MAX(id) FROM testi").fetchone()[0])
    if tid is None:
        sys.exit("[NAPAKA] V bazi ni testov.")
    r = analyse(con, tid)
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        print_report(r)


if __name__ == "__main__":
    main()
