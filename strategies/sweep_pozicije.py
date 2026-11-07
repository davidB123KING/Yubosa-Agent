"""Sweep pozicije (prevod Pine v6 -> Python, NAČIN 2: prepare + simulate_trades).

Vir: strategies/pine/sweep_pozicije.pine. Testirano na US30.cash M1.

Pravila (privzete vrednosti iz Pine):
  - Trading dan se začne ob 00:00 Europe/Ljubljana. Ob novem dnevu se nivoji pobrišejo.
  - Swingi 21/21 iz istega dneva so nivoji za entry. Sveča, ki z body-jem (close) zapre
    nad vrhom -> LONG, pod dnom -> SHORT. Entry = close te sveče.
    Nivo se porabi, tudi če trade ni odprt (npr. ker je druga pozicija še aktivna).
  - Če sta na isti sveči prebita vrh in dno -> brez trada.
  - SL = zadnja potrjena likvidnostna točka 5/5 pod ceno (long) / nad ceno (short). TP = RR x SL.
  - BE: ko je profit >= 50 % R, gre SL na entry (velja od naslednje sveče).
  - Max. trajanje 300 sveč -> zapre na close (v Pine se ta izhod ne šteje v statistiko, tu se).
  - Ena pozicija naenkrat; nova ne na sveči, kjer se je prejšnja zaprla.
  - start_mode: "HH+LL" = setupi šele, ko sta v dnevu potrjena HH in LL; "Takoj" = brez pogoja.

Časi v podatkih = FTMO strežnik = America/New_York + 7 h.
Vse cene so BID; spread in lote doda motor.
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy

P = dict(
    swLen=21, lqLeft=5, lqRight=5, rr=1.0, maxBars=300,
    useBE=True, beTrig=50.0, dayHour=0, liqCap=200,
    server_minus_ny_h=7,
)


def _pivots(x: np.ndarray, left: int, right: int, high: bool) -> np.ndarray:
    """ta.pivothigh/pivotlow(left, right): pivot strogo nad levimi in >= desnim svečam.
    Vrne indekse pivotov p (potrjeni na p + right)."""
    n = len(x)
    if n < left + right + 1:
        return np.array([], dtype=np.int64)
    y = x if high else -x
    wl = np.lib.stride_tricks.sliding_window_view(y, left).max(axis=1)   # wl[i] = max(y[i:i+left])
    wr = np.lib.stride_tricks.sliding_window_view(y, right).max(axis=1)
    p = np.arange(left, n - right)
    m = (y[p] > wl[p - left]) & (y[p] >= wr[p + 1])
    return p[m]


def _confirmed(x: np.ndarray, piv: np.ndarray, right: int) -> np.ndarray:
    """Vrednost pivota na sveči potrditve (p + right), sicer NaN."""
    out = np.full(len(x), np.nan)
    out[piv + right] = x[piv]
    return out


class SweepPozicije(Strategy):
    name = "sweep_pozicije"
    description = ("Sweep pozicije (Pine v6): body close čez swing 21/21 iz istega dneva -> vstop v smeri preboja, "
                   "SL na zadnji likvidnosti 5/5, TP 1R, BE pri 50 % R, max 300 sveč. M1.")
    addons: dict[str, bool] = {}
    param_grid = {"start_mode": ["HH+LL", "Takoj"]}
    params = P

    def prepare(self, df: pd.DataFrame):
        p = self.params
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)

        # dan v Europe/Ljubljana (z zamikom za dayHour)
        st = pd.Series(df["time"].to_numpy().astype("datetime64[ns]"))
        ny = (st - pd.Timedelta(hours=p["server_minus_ny_h"])).dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        off = (ny.dt.tz_convert("Europe/Ljubljana").dt.tz_localize(None) - st)
        off = off.ffill().bfill().fillna(pd.Timedelta(hours=-1))
        lj = st + off
        shift = 0 if p["dayHour"] == 0 else (24 - p["dayHour"])
        day = (lj + pd.Timedelta(hours=shift)).dt.normalize().to_numpy()
        new_day = np.zeros(len(df), bool)
        new_day[1:] = day[1:] != day[:-1]

        sw, lq_l, lq_r = p["swLen"], p["lqLeft"], p["lqRight"]
        return dict(
            new_day=new_day,
            ph=_confirmed(h, _pivots(h, sw, sw, True), sw),
            pl=_confirmed(l, _pivots(l, sw, sw, False), sw),
            lqh=_confirmed(h, _pivots(h, lq_l, lq_r, True), lq_r),
            lql=_confirmed(l, _pivots(l, lq_l, lq_r, False), lq_r),
        )

    def simulate_trades(self, df: pd.DataFrame, addons: dict, ctx) -> pd.DataFrame:
        p = self.params
        mode = addons.get("start_mode", "HH+LL")
        sw, rr, max_bars = p["swLen"], p["rr"], p["maxBars"]
        use_be, be_frac, cap = p["useBE"], p["beTrig"] / 100.0, p["liqCap"]

        o = df["open"].to_numpy(float).tolist()
        h = df["high"].to_numpy(float).tolist()
        l = df["low"].to_numpy(float).tolist()
        c = df["close"].to_numpy(float).tolist()
        t = df["time"].to_numpy()
        n = len(c)
        new_day = ctx["new_day"].tolist()
        ph, pl = ctx["ph"].tolist(), ctx["pl"].tolist()
        lqh, lql = ctx["lqh"].tolist(), ctx["lql"].tolist()

        def ready(hh, ll):
            return True if mode == "Takoj" else (hh and ll)

        day_start = None
        got_hh = got_ll = False
        hi_lvl, lo_lvl = [], []
        liq_hi, liq_lo = [], []
        last_ph = last_pl = None
        pos = None  # [dir, entry, sl, tp, start, risk, be, sl0]
        trades = []

        for i in range(n):
            if new_day[i]:
                day_start = i
                got_hh = got_ll = False
                hi_lvl.clear()
                lo_lvl.clear()

            # 1) odprta pozicija
            closed_now = False
            if pos is not None:
                d, e, sl, tp, s0, risk, be, sl0 = pos
                hit_tp = h[i] >= tp if d == 1 else l[i] <= tp
                hit_sl = l[i] <= sl if d == 1 else h[i] >= sl
                if hit_tp or hit_sl or i - s0 >= max_bars:
                    if hit_sl:  # SL ima prednost
                        px = min(o[i], sl) if d == 1 else max(o[i], sl)
                        reason = "be" if be else "sl"
                    elif hit_tp:
                        px = max(o[i], tp) if d == 1 else min(o[i], tp)
                        reason = "tp"
                    else:
                        px, reason = c[i], "time"
                    trades.append((t[s0], t[i], d, e, px, sl0, tp, reason))
                    pos = None
                    closed_now = True
                elif use_be and not be:
                    trig = e + d * risk * be_frac
                    if (h[i] >= trig) if d == 1 else (l[i] <= trig):
                        pos[6] = True
                        pos[2] = e

            # 2) body close čez nedotaknjen swing iz istega dneva
            touch_hi = touch_lo = None
            if hi_lvl:
                ci = c[i]
                hit = [v for v in hi_lvl if ci > v]
                if hit:
                    touch_hi = min(hit)
                    hi_lvl = [v for v in hi_lvl if not ci > v]
            if lo_lvl:
                ci = c[i]
                hit = [v for v in lo_lvl if ci < v]
                if hit:
                    touch_lo = max(hit)
                    lo_lvl = [v for v in lo_lvl if not ci < v]

            # 3) odpri pozicijo
            if (day_start is not None and ready(got_hh, got_ll) and pos is None and not closed_now
                    and (touch_hi is None) != (touch_lo is None)):
                ci = c[i]
                d = 1 if touch_hi is not None else -1
                s = None
                for v in reversed(liq_lo if d == 1 else liq_hi):
                    if (v < ci) if d == 1 else (v > ci):
                        s = v
                        break
                if s is not None:
                    risk = abs(ci - s)
                    pos = [d, ci, s, ci + d * rr * risk, i, risk, False, s]

            # 4) swingi
            v = ph[i]
            if v == v:  # ni NaN
                is_hh = last_ph is not None and v > last_ph
                if day_start is not None and i - sw >= day_start:
                    hi_lvl.append(v)
                    if is_hh:
                        got_hh = True
                last_ph = v
            v = pl[i]
            if v == v:
                is_ll = last_pl is not None and v < last_pl
                if day_start is not None and i - sw >= day_start:
                    lo_lvl.append(v)
                    if is_ll:
                        got_ll = True
                last_pl = v

            # 5) likvidnost 5/5
            v = lqh[i]
            if v == v:
                liq_hi.append(v)
                if len(liq_hi) > cap:
                    liq_hi.pop(0)
            v = lql[i]
            if v == v:
                liq_lo.append(v)
                if len(liq_lo) > cap:
                    liq_lo.pop(0)

        if pos is not None:
            d, e, sl, tp, s0, risk, be, sl0 = pos
            trades.append((t[s0], t[n - 1], d, e, c[n - 1], sl0, tp, "end"))

        cols = ["entry_time", "exit_time", "dir", "entry", "exit", "sl", "tp", "reason"]
        out = pd.DataFrame(trades, columns=cols)
        out["entry_time"] = pd.to_datetime(out["entry_time"])
        out["exit_time"] = pd.to_datetime(out["exit_time"])
        return out
