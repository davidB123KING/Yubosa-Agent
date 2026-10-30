"""Pregled prevoda htf_fvg_rb: statistika RB-jev/setupov, trade-i s časi (Europe/Ljubljana),
test pogleda v prihodnost (rezanje podatkov) in meritev hitrosti.

    .venv/Scripts/python.exe strategies/debug_htf_fvg_rb.py [--scale 66] [--combo all|base] [--n 15]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from strategies.htf_fvg_rb import HtfFvgRb  # noqa: E402


def to_lj(ts):
    s = pd.to_datetime(pd.Series(ts)) - pd.Timedelta(hours=7)
    return s.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="shift_forward") \
        .dt.tz_convert("Europe/Ljubljana").dt.strftime("%a %d.%m %H:%M")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data" / "US30.cash_M1.parquet"))
    ap.add_argument("--n", type=int, default=15, help="koliko tradov izpisati")
    ap.add_argument("--scale", type=int, default=0, help="podatke ponovi N-krat (samo meritev hitrosti)")
    args = ap.parse_args()

    df = pd.read_parquet(args.file).sort_values("time").reset_index(drop=True)
    st = HtfFvgRb()
    names = list(st.addons)
    ALL = {k: True for k in names}
    BASE = {k: False for k in names}

    if args.scale:
        parts = []
        for i in range(args.scale):
            d = df.copy()
            d["time"] = d["time"] + pd.Timedelta(days=21 * i)
            d[["open", "high", "low", "close"]] += (i % 7) * 37.0
            parts.append(d)
        big = pd.concat(parts, ignore_index=True)
        print(f"Meritev hitrosti na {len(big):,} svečah ({args.scale}× ponovljeni podatki)")
        t0 = time.time(); ctx = st.prepare(big); tp = time.time() - t0
        print(f"  prepare: {tp:.1f} s")
        combos = [dict(zip(names, b)) for b in itertools.product([False, True], repeat=len(names))]
        rng = np.random.default_rng(0)
        pick = [combos[0], combos[-1]] + [combos[i] for i in rng.choice(len(combos), 6, replace=False)]
        ts = []
        for cmb in pick:
            t0 = time.time(); tr = st.simulate_trades(big, cmb, ctx); ts.append(time.time() - t0)
            print(f"  {'+'.join(k for k, v in cmb.items() if v) or 'base':80s} {len(tr):6d} tradov  {ts[-1]:.2f} s")
        print(f"  povprečje na kombinacijo: {np.mean(ts):.2f} s, max {np.max(ts):.2f} s")
        return

    print(f"Podatki: {len(df):,} M1 sveč, {df.time.iloc[0]} -> {df.time.iloc[-1]} (čas strežnika)")
    t0 = time.time(); ctx = st.prepare(df); print(f"prepare: {time.time() - t0:.2f} s")
    print("Statistika (neodvisno od add-onov):")
    for k, v in ctx["stats"].items():
        print(f"  {k:16s} {v}")
    cd = ctx["cand"]
    print(f"  setup kandidati: long {int(cd['bull'].sum())}, short {int((~cd['bull']).sum())}")
    life = (np.minimum(cd["R_common"], cd["R_touch_v"][:, 0]) - cd["r"])
    print(f"  življenjska doba setupa (base) [1m sveč]: median {np.median(life):.0f}, max {life.max()}")
    print(f"  NWOG: {[(str(ctx['time'][w]), round(t - b, 2), str(ctx['time'][min(k, ctx['n'] - 1)])) for w, t, b, k in ctx['nwog']]}")

    # RB seznam (prvih 20)
    print("\nRB kandidati (čas zadnje 1m sveče B, Ljubljana):")
    rb = pd.DataFrame({"r_time": ctx["time"][cd["r"]], "dir": np.where(cd["bull"], "L", "S"),
                       "zBot": cd["zBot"], "zTop": cd["zTop"], "sl_pts": np.abs(cd["ePx"] - cd["sl"]),
                       "swAny": cd["swAny"], "swIn": cd["swIn"], "deliv": cd["deliv"], "star": cd["star"]})
    rb["lj"] = to_lj(rb["r_time"]).values
    print(rb.head(20).to_string(index=False))

    for label, cmb in (("BASE (vsi add-oni izklopljeni)", BASE), ("ALL (vsi add-oni = privzeti Pine)", ALL)):
        t0 = time.time(); tr = st.simulate_trades(df, cmb, ctx); dt = time.time() - t0
        print(f"\n=== {label}: {len(tr)} tradov, {dt * 1000:.0f} ms ===")
        if tr.empty:
            continue
        tr = tr.copy()
        tr["sl_pts"] = (tr["entry"] - tr["sl"]).abs()
        tr["R_bid"] = (tr["exit"] - tr["entry"]) * tr["dir"] / tr["sl_pts"]
        tr["tp_R"] = (tr["tp"] - tr["entry"]).abs() / tr["sl_pts"]
        print("reason:", tr["reason"].value_counts().to_dict(), "| dir:", tr["dir"].value_counts().to_dict())
        print("SL razdalja [točke]:", tr["sl_pts"].describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
        print("TP vir:", tr["tp_src"].value_counts().to_dict())
        print(f"vsota R (bid, brez spreada): {tr['R_bid'].sum():.2f}")
        # look-ahead sanity: vstop strogo po RB sveči in po sveči naročila
        assert (pd.to_datetime(tr["entry_time"]) > pd.to_datetime(tr["rb_time"])).all()
        show = tr.head(args.n)
        print(pd.DataFrame({
            "RB (Lj)": to_lj(show["rb_time"]).values, "vstop (Lj)": to_lj(show["entry_time"]).values,
            "izhod (Lj)": to_lj(show["exit_time"]).values, "dir": show["dir"].values,
            "entry": show["entry"].values, "sl": show["sl"].values, "tp": show["tp"].round(2).values,
            "tp_vir": show["tp_src"].values, "exit": show["exit"].round(2).values, "reason": show["reason"].values,
            "R": show["R_bid"].round(2).values}).to_string(index=False))

    # ── test pogleda v prihodnost: odreži podatke, trade-i zaprti pred rezom morajo biti enaki ──
    print("\nTest look-ahead (rezanje podatkov):")
    ok_all = True
    for frac in (0.4, 0.6, 0.8):
        cut = int(len(df) * frac)
        sub = df.iloc[:cut].reset_index(drop=True)
        cs = st.prepare(sub)
        for label, cmb in (("base", BASE), ("all", ALL)):
            full = st.simulate_trades(df, cmb, ctx)
            part = st.simulate_trades(sub, cmb, cs)
            lim = sub["time"].iloc[-1]
            f1 = full[pd.to_datetime(full["exit_time"]) < lim].reset_index(drop=True) if len(full) else full
            p1 = part[pd.to_datetime(part["exit_time"]) < lim].reset_index(drop=True) if len(part) else part
            cols = ["entry_time", "exit_time", "dir", "entry", "exit", "sl", "tp", "reason"]
            same = len(f1) == len(p1) and (len(f1) == 0 or f1[cols].equals(p1[cols]))
            ok_all &= same
            print(f"  rez {frac:.0%} {label}: polni {len(f1)} / odrezani {len(p1)} tradov -> {'OK' if same else 'RAZLIKA'}")
            if not same:
                print(f1[cols].tail(3).to_string()); print(p1[cols].tail(3).to_string())
    print("Look-ahead test:", "OK" if ok_all else "NAPAKA")

    # ── vse kombinacije: čas in št. tradov ──
    combos = [dict(zip(names, b)) for b in itertools.product([False, True], repeat=len(names))]
    t0 = time.time()
    cnt = [len(st.simulate_trades(df, cmb, ctx)) for cmb in combos]
    dt = time.time() - t0
    print(f"\n256 kombinacij: {dt:.1f} s ({dt / 256 * 1000:.0f} ms/komb.), tradov min {min(cnt)} / med {int(np.median(cnt))} / max {max(cnt)}")


if __name__ == "__main__":
    main()
