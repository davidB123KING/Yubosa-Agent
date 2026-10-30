"""Simulacija tradov in metrike. Generično – ne spreminjaj za posamezno strategijo.

Pravila izvedbe:
- signal na zaprtju bara i -> vstop na open bara i+1, ena pozicija naenkrat
- cene v podatkih so bid; long vstop = ask (open + spread), short izhod = ask
- SL in TP znotraj bara; če sta oba zadeta v istem baru, šteje SL (konzervativno)
- gap čez SL/TP: izhod na open tistega bara
- velikost lota: fiksno tveganje v valuti računa, omejeno na max_lot
"""
import math

import numpy as np
import pandas as pd

CHUNK = 512


def _first(mask_fn, start: int, n: int) -> int:
    """Prvi indeks >= start, kjer je mask_fn(slice) True. Išče po kosih (hitro)."""
    size = CHUNK
    i = start
    while i < n:
        j = min(i + size, n)
        m = mask_fn(i, j)
        if m.any():
            return i + int(np.argmax(m))
        i = j
        size *= 2
    return n


def lot_size(dist: float, meta: dict, cfg: dict) -> float:
    value_per_lot = dist / meta["tick_size"] * meta["tick_value"]
    if value_per_lot <= 0:
        return 0.0
    raw = cfg["risk_per_trade"] / value_per_lot
    step = meta.get("volume_step", 0.01) or 0.01
    lots = math.floor(raw / step + 1e-9) * step
    lots = min(lots, cfg["max_lot"], meta.get("volume_max", 1e9))
    return round(max(lots, meta.get("volume_min", step)), 8)


def simulate(df: pd.DataFrame, sig: pd.DataFrame, meta: dict, cfg: dict, symbol: str) -> pd.DataFrame:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = df["time"].to_numpy()
    n = len(df)

    if cfg.get("spread_mode") == "fixed":
        pts = cfg.get("fixed_spread_points", {}).get(symbol)
        if pts is None:
            raise ValueError(f"spread_mode=fixed, a za {symbol} ni vrednosti v fixed_spread_points")
        spread = np.full(n, pts * meta["point"])
    else:
        spread = df["spread"].to_numpy(float) * meta["point"]

    signal = sig["signal"].fillna(0).to_numpy(int)
    sl_arr = sig["sl"].to_numpy(float)
    tp_arr = sig["tp"].to_numpy(float) if "tp" in sig else np.full(n, np.nan)
    ex_arr = sig["exit"].fillna(False).to_numpy(bool) if "exit" in sig else None
    ex_idx = np.flatnonzero(ex_arr) if ex_arr is not None else np.array([], int)

    tv, ts = meta["tick_value"], meta["tick_size"]
    comm = cfg.get("commission_per_lot", 0.0)
    trades = []
    free_from = 0

    for s in np.flatnonzero(signal != 0):
        e = s + 1
        if s < free_from or e >= n:
            continue
        d = signal[s]
        sl, tp = sl_arr[s], tp_arr[s]
        if np.isnan(sl):
            continue

        if d == 1:
            entry = o[e] + spread[e]
            dist = entry - sl
            if dist <= 0:
                continue
            hit_sl = lambda i, j: l[i:j] <= sl
            hit_tp = (lambda i, j: h[i:j] >= tp) if not np.isnan(tp) else None
        else:
            entry = o[e]
            dist = sl - entry
            if dist <= 0:
                continue
            hit_sl = lambda i, j: h[i:j] + spread[i:j] >= sl
            hit_tp = (lambda i, j: l[i:j] + spread[i:j] <= tp) if not np.isnan(tp) else None

        j_sl = _first(hit_sl, e, n)
        j_tp = _first(hit_tp, e, n) if hit_tp else n
        j_ex = n
        if len(ex_idx):
            k = np.searchsorted(ex_idx, e)
            if k < len(ex_idx) and ex_idx[k] + 1 < n:
                j_ex = ex_idx[k] + 1  # izhod na open naslednjega bara

        j = min(j_sl, j_tp, j_ex)
        if j >= n:
            j, reason = n - 1, "end"
            px = c[j] if d == 1 else c[j] + spread[j]
        elif j == j_sl:  # SL ima prednost pri izenačenju
            reason = "sl"
            if j > e:  # gap
                px = min(o[j], sl) if d == 1 else max(o[j] + spread[j], sl)
            else:
                px = sl
        elif j == j_tp:
            reason = "tp"
            if j > e:
                px = max(o[j], tp) if d == 1 else min(o[j] + spread[j], tp)
            else:
                px = tp
        else:
            reason = "exit"
            px = o[j] if d == 1 else o[j] + spread[j]

        lots = lot_size(dist, meta, cfg)
        if lots <= 0:
            continue
        pnl = (px - entry) * d / ts * tv * lots - comm * lots
        trades.append({
            "entry_time": t[e], "exit_time": t[j], "dir": int(d), "entry": entry, "exit": px,
            "sl": sl, "tp": tp, "lots": lots, "risk": dist / ts * tv * lots,
            "pnl": pnl, "r": pnl / cfg["risk_per_trade"], "reason": reason,
        })
        free_from = j  # nov signal šele od bara izhoda naprej (vstop na naslednjem open)

    return pd.DataFrame(trades)


def price_trades(raw: pd.DataFrame, df: pd.DataFrame, meta: dict, cfg: dict, symbol: str) -> pd.DataFrame:
    """Za strategije, ki same simulirajo trade-e (Strategy.simulate_trades).

    raw: entry_time, exit_time, dir (1/-1), entry, exit, sl (ZAČETNI SL), reason – vse na bid cenah.
    Motor doda: velikost lota (fiksni risk na začetni SL, max_lot), spread (en spread na trade,
    vzet na sveči vstopa – long ga plača ob vstopu, short ob izhodu), provizijo, pnl in R.
    """
    cols = ["entry_time", "exit_time", "dir", "entry", "exit", "sl", "tp", "lots", "risk", "pnl", "r", "reason"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols)
    tr = raw.copy().reset_index(drop=True)
    if "tp" not in tr:
        tr["tp"] = np.nan

    if cfg.get("spread_mode") == "fixed":
        sp = np.full(len(tr), cfg["fixed_spread_points"][symbol] * meta["point"])
    else:
        sp_series = pd.Series(df["spread"].to_numpy(float) * meta["point"], index=df["time"])
        sp_series = sp_series[~sp_series.index.duplicated()]
        sp = sp_series.reindex(pd.DatetimeIndex(tr["entry_time"]), method="ffill").fillna(0).to_numpy()

    tv, ts = meta["tick_value"], meta["tick_size"]
    comm = cfg.get("commission_per_lot", 0.0)
    d = tr["dir"].to_numpy(int)
    entry, exit_, sl = tr["entry"].to_numpy(float), tr["exit"].to_numpy(float), tr["sl"].to_numpy(float)
    dist = np.abs(entry - sl) + sp  # dejanska razdalja do SL vključno s spreadom
    lots = np.array([lot_size(x, meta, cfg) if x > 0 else 0.0 for x in dist])
    move = (exit_ - entry) * d - sp
    tr["lots"] = lots
    tr["risk"] = dist / ts * tv * lots
    tr["pnl"] = move / ts * tv * lots - comm * lots
    tr["r"] = tr["pnl"] / cfg["risk_per_trade"]
    tr = tr[lots > 0]
    return tr[cols].reset_index(drop=True)


def metrics(tr: pd.DataFrame, balance: float) -> dict:
    if tr.empty:
        return dict(st_tradov=0, win_rate=0.0, profit_factor=0.0, expectancy_r=0.0,
                    max_dd=0.0, max_dd_pct=0.0, net_profit=0.0)
    pnl = tr["pnl"].to_numpy()
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = balance + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[balance], eq]))[1:]
    dd = peak - eq
    i = int(np.argmax(dd))
    return dict(
        st_tradov=int(len(tr)),
        win_rate=float((pnl > 0).mean()),
        profit_factor=float(gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0),
        expectancy_r=float(tr["r"].mean()),
        max_dd=float(dd.max()),
        max_dd_pct=float(dd[i] / peak[i] * 100) if dd.max() > 0 else 0.0,
        net_profit=float(pnl.sum()),
    )
