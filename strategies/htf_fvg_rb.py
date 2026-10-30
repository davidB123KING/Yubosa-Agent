"""HTF FVG + 15m RB Strategy (prevod Pine v6 -> Python, NAČIN 2: prepare + simulate_trades).

Vir: strategies/pine/htf_fvg_rb.pine. Strategija je za 1-minutni graf (US30.cash M1).

Struktura:
  prepare(df)        – vse, kar NI odvisno od add-onov (HTF FVG-ji, pivoti, sweepi, HH/LL, seje,
                       NWOG, NY/Asia bias, RB kandidati z vsemi zastavicami, življenjske dobe setupov).
  simulate_trades()  – na kombinacijo: izbor RB-jev (numpy), čakajoči limit order na svečo (numpy),
                       Python zanka samo po odprtih pozicijah (BE / alt BE / trailing).

Časi v podatkih = FTMO strežnik = America/New_York + 7 h. Seje so v Europe/Ljubljana.
Vse cene so BID (kot graf v TradingView); spread in lote doda motor.
"""
import numpy as np
import pandas as pd

from strategies.base import Strategy

# ──────────────────────────────────────────────────────────────────────────────
# Fiksni parametri (privzete vrednosti iz Pine)
# ──────────────────────────────────────────────────────────────────────────────
P = dict(
    minFVG=5.0, fvg_caps={"15m": 30, "1h": 20, "4h": 15, "1D": 15},
    deliv_tfs=("15m", "1h", "4h"),            # useDelivD = false
    reqStandalone=True,                       # "Beyond previous extreme"
    sweepTolerance=1.0, pivotLenLiq=5,
    starWickRatio=1.0,
    fibLevel=0.786, reqPullback=False,
    tpRR=4.0, minSessRR=1.0, useSessTP=True, useRunSess=True,
    hhLen=21, trailPivLen=21, trailBuffer=0.0, trailStartR=0.0, useAltBE=True,
    session=(9 * 60, 22 * 60), blackout=(15 * 60 + 25, 15 * 60 + 30), cancelAtSessEnd=True,
    sess_levels={"Asia": (3 * 60, 7 * 60), "London": (9 * 60, 12 * 60),
                 "NY AM": (15 * 60 + 30, 18 * 60), "NY PM": (20 * 60 + 30, 22 * 60)},
    nyTrack=(12 * 60, 18 * 60), nyWindow=(15 * 60 + 30, 18 * 60),
    asiaWindow=(7 * 60, 12 * 60),
    nwogIgnoreMin=15, nwogMinGap=0.0,         # nwogMode = "Samo short"
    server_minus_ny_h=7,
)


# ──────────────────────────────────────────────────────────────────────────────
# Pomožno: vektorsko iskanje "prvi indeks k >= start, kjer x[k] >= L (ali > L)"
# Bloki po 64 + sparse table nad maksimumi blokov -> O(log n) na poizvedbo, vektorsko.
# ──────────────────────────────────────────────────────────────────────────────
class _Cross:
    B = 64

    def __init__(self, x: np.ndarray):
        n = len(x)
        nb = (n + self.B - 1) // self.B
        xp = np.full(nb * self.B, -np.inf)
        xp[:n] = x
        self.n, self.nb = n, nb
        self.rows = xp.reshape(nb, self.B)
        bm = self.rows.max(axis=1)
        st = [bm]
        j = 1
        while (1 << (j - 1)) < nb:
            prev, h = st[-1], 1 << (j - 1)
            cur = prev.copy()
            cur[: nb - h] = np.maximum(prev[: nb - h], prev[h:])
            st.append(cur)
            j += 1
        self.st = st

    def first(self, starts, levels, strict=False, chunk=20000):
        starts = np.asarray(starts, dtype=np.int64)
        levels = np.asarray(levels, dtype=float)
        n, nb, B = self.n, self.nb, self.B
        out = np.full(len(starts), n, dtype=np.int64)
        ar = np.arange(B)
        for c0 in range(0, len(starts), chunk):
            s = starts[c0:c0 + chunk]
            L = levels[c0:c0 + chunk]
            ok = (s < n) & (s >= 0) & ~np.isnan(L)
            sc = np.clip(s, 0, max(n - 1, 0))
            b0 = sc // B
            row = self.rows[b0]
            cond = (row > L[:, None]) if strict else (row >= L[:, None])
            cond &= ar[None, :] >= (sc - b0 * B)[:, None]
            has = cond.any(axis=1) & ok
            res = np.where(has, b0 * B + cond.argmax(axis=1), n)
            need = ok & ~has
            pos = b0 + 1
            for j in range(len(self.st) - 1, -1, -1):
                step = 1 << j
                can = need & (pos + step <= nb)
                bmax = self.st[j][np.minimum(pos, nb - 1)]
                nf = (bmax <= L) if strict else (bmax < L)
                pos = np.where(can & nf, pos + step, pos)
            inb = need & (pos < nb)
            pc = np.minimum(pos, nb - 1)
            row2 = self.rows[pc]
            cond2 = (row2 > L[:, None]) if strict else (row2 >= L[:, None])
            fnd = inb & cond2.any(axis=1)
            res = np.where(fnd, pc * B + cond2.argmax(axis=1), res)
            out[c0:c0 + chunk] = np.minimum(res, n)
        return out


def _pivots(x: np.ndarray, ln: int, high: bool):
    """ta.pivothigh/pivotlow(ln, ln). Predpostavka glede izenačenj: pivot mora biti strogo nad
    levimi in >= desnim svečam (pri enakih vrhovih je pivot levi). Vrne indekse pivotov p
    (potrjeni na p+ln)."""
    n = len(x)
    if n < 2 * ln + 1:
        return np.array([], dtype=np.int64)
    y = x if high else -x
    w = np.lib.stride_tricks.sliding_window_view(y, ln).max(axis=1)  # w[i] = max(y[i:i+ln])
    p = np.arange(ln, n - ln)
    left = w[p - ln]
    right = w[p + 1]
    m = (y[p] > left) & (y[p] >= right)
    return p[m]


def _runs(mask: np.ndarray):
    """Začetki in konci (vključno) zaporednih True v maski."""
    m = mask.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def _in_win(mins: np.ndarray, w):
    a, b = w
    return (mins >= a) & (mins < b) if a <= b else (mins >= a) | (mins < b)


class HtfFvgRb(Strategy):
    name = "htf_fvg_rb"
    description = ("HTF FVG + 15m Rejection Block (Pine v6): limit vstop na robu RB cone, SL na fiksni točki, "
                   "TP na HH/LL ali session high/low (sicer 4R), BE po pobrani likvidnosti, trailing na HL/LH. M1.")
    addons = {
        "fvg_delivery": False,       # reqFVGDelivery
        "sweep_inside_rb": False,    # reqSweepInsideRB (False -> še vedno reqLiqSweep = poljuben sweep)
        "star_hammer_filter": False, # filterStarBear + filterHammerBull
        "invalidation": False,       # reqInvalidation
        "ny_sweep_filter": False,    # useNYFilter
        "asia_sweep_filter": False,  # useAsiaFilter
        "nwog_filter": False,        # useNWOG
        "trailing_sl": False,        # useTrail
    }
    # Izvedba na MT5 prek webhooka: lot iz originalnega SL (signal), SL dejansko odmaknjen
    # za sl_buffer točk, BE postavljen be_offset točk v minus.
    param_grid = {
        "sl_buffer": [0, 1, 2, 3, 4, 5],
        "be_offset": [0, 1, 2, 3, 4, 5],
    }
    params = P

    # ══════════════════════════════════════════════════════════════════════
    # PREPARE (enkrat na par)
    # ══════════════════════════════════════════════════════════════════════
    def prepare(self, df: pd.DataFrame):
        p = self.params
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        n = len(df)
        tmin = df["time"].to_numpy().astype("datetime64[m]").astype(np.int64)  # strežnik, minute

        # ── čas v Europe/Ljubljana ──
        st = pd.Series(df["time"].to_numpy().astype("datetime64[ns]"))
        ny = (st - pd.Timedelta(hours=p["server_minus_ny_h"])).dt.tz_localize(
            "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
        lj = ny.dt.tz_convert("Europe/Ljubljana")
        off = (lj.dt.tz_localize(None) - st).dt.total_seconds() / 60.0
        off = off.ffill().bfill().fillna(-60).to_numpy()
        ljm = ((tmin + off.astype(np.int64)) % 1440).astype(np.int32)   # minuta dneva (Lj), čas odprtja
        ljm_close = (ljm + 1) % 1440                                     # time_close 1m sveče

        in_sess = _in_win(ljm, p["session"])
        bo_now = _in_win(ljm, p["blackout"])
        bo_next = _in_win(ljm_close, p["blackout"])

        cH, cNL, cC, cNC = _Cross(h), _Cross(-l), _Cross(c), _Cross(-c)

        # ── 1m likvidnost (pivot 5) in sweep dogodki ──
        L5 = p["pivotLenLiq"]
        ph5 = _pivots(h, L5, True)
        pl5 = _pivots(l, L5, False)
        swH = cH.first(ph5 + L5, h[ph5], strict=False)          # high >= level, od potrditve naprej
        swL = cNL.first(pl5 + L5, -l[pl5], strict=False)        # low <= level
        mH, mL = swH < n, swL < n
        evH_bar, evH_lvl, evH_piv = swH[mH], h[ph5][mH], ph5[mH]
        evL_bar, evL_lvl, evL_piv = swL[mL], l[pl5][mL], pl5[mL]
        oh, ol = np.argsort(evH_bar, kind="stable"), np.argsort(evL_bar, kind="stable")
        evH_bar, evH_lvl, evH_piv = evH_bar[oh], evH_lvl[oh], evH_piv[oh]
        evL_bar, evL_lvl, evL_piv = evL_bar[ol], evL_lvl[ol], evL_piv[ol]
        ptrH = np.searchsorted(evH_bar, np.arange(n + 1)).astype(np.int64)
        ptrL = np.searchsorted(evL_bar, np.arange(n + 1)).astype(np.int64)

        # ── HH / LL (pivot 21) in HL / LH za trailing ──
        L21 = p["hhLen"]
        ph21 = _pivots(h, L21, True)
        pl21 = _pivots(l, L21, False)
        vH, vL = h[ph21], l[pl21]
        prevH = np.concatenate([[np.nan], vH[:-1]])
        prevL = np.concatenate([[np.nan], vL[:-1]])
        isHH = vH > prevH            # NaN -> False
        isLL = vL < prevL
        hh_c = ph21[isHH] + L21
        hh_lvl = vH[isHH]
        hh_rem = cH.first(hh_c, hh_lvl, strict=True)            # high > level -> odstranjen
        ll_c = pl21[isLL] + L21
        ll_lvl = vL[isLL]
        ll_rem = cNL.first(ll_c, -ll_lvl, strict=True)          # low < level
        TL = p["trailPivLen"]
        if TL == L21:
            tph, tpl = ph21, pl21
        else:
            tph, tpl = _pivots(h, TL, True), _pivots(l, TL, False)
        tvH, tvL = h[tph], l[tpl]
        isLH = tvH < np.concatenate([[np.nan], tvH[:-1]])
        isHL = tvL > np.concatenate([[np.nan], tvL[:-1]])
        hl_lvl = np.full(n, np.nan)
        hl_lvl[tpl[isHL] + TL] = tvL[isHL]
        lh_lvl = np.full(n, np.nan)
        lh_lvl[tph[isLH] + TL] = tvH[isLH]

        # ── Session high / low ──
        sess = {}
        for nm, w in p["sess_levels"].items():
            inS = _in_win(ljm, w)
            rs, re_ = _runs(inS)
            hi = np.array([h[a:b + 1].max() for a, b in zip(rs, re_)])
            lo = np.array([l[a:b + 1].min() for a, b in zip(rs, re_)])
            e1 = re_ + 1
            okE = e1 < n
            de1, dhi, dlo = e1[okE], hi[okE], lo[okE]
            hi_rem = cH.first(de1 + 1, dhi, strict=True)
            lo_rem = cNL.first(de1 + 1, -dlo, strict=True)
            sess[nm] = dict(inS=inS, rs=rs, re=re_, e1=de1, hi=dhi, lo=dlo, hi_rem=hi_rem, lo_rem=lo_rem)

        def done_at(rec, j):
            k = np.searchsorted(rec["e1"], j, side="right") - 1
            if k < 0:
                return np.nan, np.nan
            hi = rec["hi"][k] if j < rec["hi_rem"][k] else np.nan
            lo = rec["lo"][k] if j < rec["lo_rem"][k] else np.nan
            return hi, lo

        # ── NY sweep filter (London H/L) in Asia sweep filter (Asia H/L) ──
        def window_bias(win, ref, low_first):
            inW = _in_win(ljm, win)
            bias = np.zeros(n, dtype=np.int8)
            for a, b in zip(*_runs(inW)):
                hi, lo = done_at(sess[ref], a)
                kH = cH.first([a], [hi], strict=True)[0] if not np.isnan(hi) else n
                kL = cNL.first([a], [-lo], strict=True)[0] if not np.isnan(lo) else n
                evs = []
                if kL <= b:
                    evs.append((kL, 0 if low_first else 1, 1))
                if kH <= b:
                    evs.append((kH, 1 if low_first else 0, -1))
                for k, _, v in sorted(evs):
                    bias[k:b + 1] = v
            return bias, inW

        nyb, _ = window_bias(p["nyTrack"], "London", low_first=True)     # Pine: low check, nato high
        in_ny = _in_win(ljm, p["nyWindow"])
        asb, _ = window_bias(p["asiaWindow"], "Asia", low_first=False)   # Pine: high check, nato low

        # ── NWOG ──
        days = tmin // 1440
        wk = (days + 3) // 7                       # teden od ponedeljka (čas strežnika)
        nw = np.flatnonzero(np.diff(wk) != 0) + 1
        nwp = np.zeros(n, dtype=bool)
        nwog_list = []
        for i, w0 in enumerate(nw):
            w1 = nw[i + 1] if i + 1 < len(nw) else n
            pc = c[w0 - 1]
            gT, gB = max(o[w0], pc), min(o[w0], pc)
            if gT - gB > 0 and gT - gB >= p["nwogMinGap"]:
                k0 = np.searchsorted(tmin, tmin[w0] + p["nwogIgnoreMin"], side="left")
                k0 = max(k0, w0 + 1)
                seg = np.arange(k0, w1)
                hit = seg[(h[seg] >= gB) & (l[seg] <= gT)] if len(seg) else seg
                k = hit[0] if len(hit) else w1
                nwp[w0:k] = True
                nwog_list.append((w0, gT, gB, k))
        # pending se prenese čez teden, če ni dotika? Pine: newWeek -> pending := false, torej ne.

        # ── HTF sveče (strežniški čas; 4h poravnano na 00/04/08.. strežnika, D = dan strežnika) ──
        def htf(minutes):
            key = tmin // minutes
            s0 = np.concatenate([[0], np.flatnonzero(np.diff(key) != 0) + 1])
            e0 = np.concatenate([s0[1:], [n]])
            return dict(s=s0, e=e0, o=o[s0], h=np.maximum.reduceat(h, s0), l=np.minimum.reduceat(l, s0),
                        c=c[e0 - 1], t=key[s0] * minutes)

        tf = {"15m": htf(15), "1h": htf(60), "4h": htf(240), "1D": htf(1440)}
        q15 = tf["15m"]
        nq = len(q15["s"])

        # FVG kandidati na HTF baru k: sveče k-3, k-2, k-1 (vse zaprte) ; mitigacija s close[k-1]
        for name, T in tf.items():
            K = len(T["s"])
            ct = np.full(K, np.nan); cb = np.full(K, np.nan); cd = np.zeros(K, np.int8)
            if K > 3:
                k = np.arange(3, K)
                bull = T["l"][k - 1] > T["h"][k - 3]
                bear = T["h"][k - 1] < T["l"][k - 3]
                top = np.where(bull, T["l"][k - 1], np.where(bear, T["l"][k - 3], np.nan))
                bot = np.where(bull, T["h"][k - 3], np.where(bear, T["h"][k - 1], np.nan))
                okz = (bull | bear) & ((top - bot) >= p["minFVG"])
                ct[k] = np.where(okz, top, np.nan)
                cb[k] = np.where(okz, bot, np.nan)
                cd[k] = np.where(okz, np.where(bull, 1, -1), 0)
            pcl = np.concatenate([[np.nan], T["c"][:-1]])
            T.update(ct=ct, cb=cb, cd=cd, pc=pcl)
            T["start_of"] = {int(s): kk for kk, s in enumerate(T["s"])} if name != "15m" else None

        # ── 15m RB kandidati (vektorsko) ──
        qs, qe = q15["s"], q15["e"]
        O15, H15, L15, C15 = q15["o"], q15["h"], q15["l"], q15["c"]
        r_all = qe - 1
        conf = (tmin[r_all] % 15 == 14)                     # zadnja 1m sveča 15m sveče obstaja
        qq = np.arange(nq)
        okq = conf & (qq >= 2)
        A, Bq, Z = np.clip(qq - 1, 0, None), qq, np.clip(qq - 2, 0, None)
        rawBull = (C15[A] < O15[A]) & (C15[Bq] > O15[Bq])
        rawBear = (C15[A] > O15[A]) & (C15[Bq] < O15[Bq])
        standOK = np.where(rawBear, H15[Bq] > H15[Z], L15[Bq] < L15[Z])
        if not p["reqStandalone"]:
            standOK[:] = True
        insessRB = _in_win(ljm_close[r_all], p["session"])
        rbq = okq & (rawBull | rawBear) & standOK & insessRB
        n_raw_pattern = int((okq & (rawBull | rawBear)).sum())

        # ── FVG zanka (stanje FVG seznamov na RB svečah) ──
        caps = p["fvg_caps"]
        lists = {k: [] for k in tf}       # elementi: [top, bot, bull(bool)]
        pend = {k: None for k in tf}      # (top, bot, bull, next_i, end_i)
        deliv = np.zeros(nq, dtype=bool)
        biasok = np.ones(nq, dtype=bool)
        tol = 0.02  # syminfo.mintick * 2 (US30 mintick = 0.01)

        def exists(lst, top, bot, bull):
            for it in lst:
                if it[2] == bull and abs(it[0] - top) < tol and abs(it[1] - bot) < tol:
                    return True
            return False

        def cap(lst, mx, px):
            while len(lst) > mx:
                wi, wd = -1, -1.0
                for ii, it in enumerate(lst):
                    d = px - it[0] if px > it[0] else (it[1] - px if px < it[1] else 0.0)
                    if d > wd:
                        wd, wi = d, ii
                del lst[wi]

        def process(name, k, i0):
            T = tf[name]
            lst = lists[name]
            d = T["cd"][k]
            if d != 0:
                top, bot, bull = T["ct"][k], T["cb"][k], bool(d == 1)
                if not exists(lst, top, bot, bull):
                    lst.append([top, bot, bull])
            pc = T["pc"][k]
            if not np.isnan(pc):
                lst[:] = [it for it in lst if not ((it[2] and pc < it[1]) or ((not it[2]) and pc > it[0]))]
            cap(lst, caps[name], c[i0])
            pend[name] = None
            if d != 0 and not exists(lst, top, bot, bull):
                pend[name] = (top, bot, bull, i0 + 1, int(T["e"][k]))

        def readd(name, upto):
            pd_ = pend[name]
            if pd_ is None:
                return
            top, bot, bull, i, end_i = pd_
            lst = lists[name]
            last = min(upto, end_i)
            while i < last:
                if not exists(lst, top, bot, bull):
                    lst.append([top, bot, bull])
                cap(lst, caps[name], c[i])
                i += 1
                if exists(lst, top, bot, bull):
                    pend[name] = None
                    return
            pend[name] = None if i >= end_i else (top, bot, bull, i, end_i)

        hiTF = ("1h", "4h", "1D")
        for q in range(nq):
            i0, i1 = int(qs[q]), int(qe[q])
            process("15m", q, i0)
            for name in hiTF:
                k = tf[name]["start_of"].get(i0)
                if k is not None:
                    process(name, k, i0)
            for name in tf:
                if pend[name] is not None:
                    readd(name, i1)
            if rbq[q]:
                a, b = q - 1, q
                if rawBull[q]:
                    isB = True
                    zTop = max(min(O15[a], C15[a]), min(O15[b], C15[b]))
                    zBot = min(L15[a], L15[b])
                else:
                    isB = False
                    zTop = max(H15[a], H15[b])
                    zBot = min(max(O15[a], C15[a]), max(O15[b], C15[b]))
                dv = False
                for name in p["deliv_tfs"]:
                    for top, bot, bull in lists[name]:
                        if bull != isB:
                            continue
                        aT = H15[a] >= bot and L15[a] <= top
                        bT = H15[b] >= bot and L15[b] <= top
                        aD = C15[a] < bot if isB else C15[a] > top
                        bD = C15[b] < bot if isB else C15[b] > top
                        if (aT or bT) and not aD and not bD:
                            dv = True
                            break
                    if dv:
                        break
                deliv[q] = dv
                for top, bot, bull in lists["1D"]:
                    if bull != isB and zTop >= bot and zBot <= top:
                        biasok[q] = False
                        break

        # ── atributi RB kandidatov ──
        cq = np.flatnonzero(rbq)
        a, b = cq - 1, cq
        r = r_all[cq]
        isBull = rawBull[cq]
        zTop = np.where(isBull, np.maximum(np.minimum(O15[a], C15[a]), np.minimum(O15[b], C15[b])),
                        np.maximum(H15[a], H15[b]))
        zBot = np.where(isBull, np.minimum(L15[a], L15[b]),
                        np.minimum(np.maximum(O15[a], C15[a]), np.maximum(O15[b], C15[b])))
        fixedA = np.where(isBull, zBot, zTop)
        ext0 = np.where(isBull, H15[b], L15[b])
        inval = np.where(isBull, np.maximum(H15[a], H15[b]), np.minimum(L15[a], L15[b]))
        fib = p["fibLevel"]
        lvl0 = ext0 + fib * (fixedA - ext0)
        valid = (lvl0 >= zBot) & (lvl0 <= zTop)
        ePx = np.where(isBull, zTop, zBot)
        sPts = np.abs(ePx - fixedA)
        w0 = qs[a]                                   # okno A+B: [prvi 1m bar A, r]

        swAny = np.zeros(len(cq), bool)
        swIn = np.zeros(len(cq), bool)
        star = np.zeros(len(cq), bool)
        tolS = p["sweepTolerance"]
        swr = p["starWickRatio"]
        for i in range(len(cq)):
            lo_, hi_ = int(w0[i]), int(r[i])
            if isBull[i]:
                ea, eb = ptrL[lo_], ptrL[hi_ + 1]
                lv = evL_lvl[ea:eb]
            else:
                ea, eb = ptrH[lo_], ptrH[hi_ + 1]
                lv = evH_lvl[ea:eb]
            swAny[i] = eb > ea
            swIn[i] = bool(((lv >= zBot[i] - tolS) & (lv <= zTop[i] + tolS)).any()) if eb > ea else False
            # shooting star (bear) / hammer (bull) na ekstremu A+B; pri izenačenju zadnja sveča
            if isBull[i]:
                seg = l[lo_:hi_ + 1]
                bi = lo_ + len(seg) - 1 - int(np.argmin(seg[::-1]))
            else:
                seg = h[lo_:hi_ + 1]
                bi = lo_ + len(seg) - 1 - int(np.argmax(seg[::-1]))
            oo, hh_, ll_, cc = o[bi], h[bi], l[bi], c[bi]
            body = abs(cc - oo)
            upW = hh_ - max(oo, cc)
            loW = min(oo, cc) - ll_
            star[i] = (loW > body * swr and upW < body) if isBull[i] else (upW > body * swr and loW < body)

        # Kandidati, ki so neodvisno od add-onov sploh lahko trade: daily bias OK, fib nivo veljaven, sPts > 0.
        keep = biasok[cq] & valid & (sPts > 0)
        stats = dict(n_15m=int(nq), n_raw_pattern=n_raw_pattern, n_rb_pre=int(len(cq)),
                     n_bias_blocked=int((~biasok[cq]).sum()), n_invalid_fib=int((~valid).sum()),
                     n_cand=int(keep.sum()), n_swAny=int(swAny[keep].sum()), n_swIn=int(swIn[keep].sum()),
                     n_deliv=int(deliv[cq][keep].sum()), n_star=int(star[keep].sum()),
                     n_liq_ph=int(len(ph5)), n_liq_pl=int(len(pl5)), n_hh=int(len(hh_c)), n_ll=int(len(ll_c)),
                     n_hl=int(isHL.sum()), n_lh=int(isLH.sum()), n_nwog=len(nwog_list))

        sel = np.flatnonzero(keep)
        cand = dict(q=cq[sel], r=r[sel], bull=isBull[sel], zTop=zTop[sel], zBot=zBot[sel], sl=fixedA[sel],
                    ePx=ePx[sel], inval=inval[sel], swAny=swAny[sel], swIn=swIn[sel], deliv=deliv[cq][sel],
                    star=star[sel])
        m = len(sel)
        rr, bl = cand["r"], cand["bull"]
        s1 = rr + 1
        # ── življenjska doba setupa (neodvisno od kombinacije) ──
        nxt_out = np.minimum.accumulate(np.where(~in_sess, np.arange(n), n)[::-1])[::-1]
        nxt_out = np.concatenate([nxt_out, [n]])
        R_sess = nxt_out[np.minimum(s1, n)]
        thr = np.where(bl, (cand["zTop"] - fib * cand["zBot"]) / (1 - fib),
                       (cand["zBot"] - fib * cand["zTop"]) / (1 - fib))
        R_brk = np.where(bl, cNC.first(s1, -cand["zBot"], strict=True), cC.first(s1, cand["zTop"], strict=True))
        R_ext = np.where(bl, cH.first(s1, thr, strict=True), cNL.first(s1, -thr, strict=True))
        R_tch = np.where(bl, cNL.first(s1, -cand["zTop"], strict=False), cH.first(s1, cand["zBot"], strict=False))
        R_inv = np.where(bl, cH.first(s1, cand["inval"], strict=True), cNL.first(s1, -cand["inval"], strict=True))
        R_common = np.minimum(np.minimum(R_sess, R_brk), R_ext)

        # dotik med blackoutom ne odstrani setupa, če je smer dovoljena (ni ne vstopa ne preklica)
        R_touch_v = np.repeat(R_tch[:, None], 8, axis=1)
        for v in range(8):
            asia, nyf, nwg = bool(v & 1), bool(v & 2), bool(v & 4)
            for i in np.flatnonzero((R_tch < R_common) & bo_now[np.minimum(R_tch, n - 1)] & (R_tch < n)):
                k = int(R_tch[i])
                while k < n and k < R_common[i] and bo_now[k]:
                    if bl[i]:
                        allowed = not ((asia and asb[k] == -1) or (nyf and in_ny[k] and nyb[k] == -1))
                    else:
                        allowed = not ((asia and asb[k] == 1) or (nyf and in_ny[k] and nyb[k] == 1) or (nwg and nwp[k]))
                    if not allowed:
                        break
                    if bl[i]:
                        k = int(cNL.first([k + 1], [-cand["zTop"][i]])[0])
                    else:
                        k = int(cH.first([k + 1], [cand["zBot"][i]])[0])
                R_touch_v[i, v] = k
        cand.update(R_common=R_common, R_touch_v=R_touch_v, R_inv=R_inv, R_sess=R_sess, R_brk=R_brk,
                    R_ext=R_ext, R_tch=R_tch)

        return dict(n=n, o=o, h=h, l=l, c=c, time=df["time"].to_numpy(), ljm=ljm,
                    in_sess=in_sess, bo_next=bo_next, asb=asb, nyb=nyb, in_ny=in_ny, nwp=nwp,
                    cand=cand, m=m, cH=cH, cNL=cNL,
                    ptrH=ptrH.tolist(), lvH=evH_lvl.tolist(), pvH=evH_piv.tolist(),
                    ptrL=ptrL.tolist(), lvL=evL_lvl.tolist(), pvL=evL_piv.tolist(),
                    hl=hl_lvl, lh=lh_lvl,
                    hh_c=hh_c, hh_lvl=hh_lvl, hh_rem=hh_rem, ll_c=ll_c, ll_lvl=ll_lvl, ll_rem=ll_rem,
                    sess=sess, stats=stats, nwog=nwog_list)

    # ══════════════════════════════════════════════════════════════════════
    # TP: f_targets (stanje na sveči j, kjer se postavi limit order)
    # ══════════════════════════════════════════════════════════════════════
    def _targets(self, X, bull, e, stop, j):
        p = self.params
        if bull:
            k = np.searchsorted(X["hh_c"], j, side="right")
            lv, rm = X["hh_lvl"][:k], X["hh_rem"][:k]
            mm = (rm > j) & (lv > e)
            swing = lv[mm].min() if mm.any() else np.nan
        else:
            k = np.searchsorted(X["ll_c"], j, side="right")
            lv, rm = X["ll_lvl"][:k], X["ll_rem"][:k]
            mm = (rm > j) & (lv < e)
            swing = lv[mm].max() if mm.any() else np.nan
        best, bname = np.nan, ""
        for nm, rec in X["sess"].items():
            vals = []
            kk = np.searchsorted(rec["e1"], j, side="right") - 1
            if kk >= 0:
                if bull and j < rec["hi_rem"][kk]:
                    vals.append((rec["hi"][kk], nm))
                if (not bull) and j < rec["lo_rem"][kk]:
                    vals.append((rec["lo"][kk], nm))
            if p["useRunSess"] and rec["inS"][j]:
                k2 = np.searchsorted(rec["rs"], j, side="right") - 1
                a = rec["rs"][k2]
                v = X["h"][a:j + 1].max() if bull else X["l"][a:j + 1].min()
                vals.append((v, nm + " (tekoči)"))
            for v, vn in vals:
                if (v > e if bull else v < e) and (np.isnan(best) or (v < best if bull else v > best)):
                    best, bname = v, vn
        sessOk = p["useSessTP"] and not np.isnan(best) and abs(best - e) >= stop * p["minSessRR"]
        if not np.isnan(swing):
            tp, src = swing, "HH" if bull else "LL"
            if sessOk and (best < swing if bull else best > swing):
                tp, src = best, bname
        elif sessOk:
            tp, src = best, bname
        else:
            tp = e + stop * p["tpRR"] if bull else e - stop * p["tpRR"]
            src = f"{p['tpRR']}R"
        return float(tp), src

    # ══════════════════════════════════════════════════════════════════════
    # Vodenje pozicije (BE, alt BE, trailing) – normaliziran prostor: short = negirane cene
    # ══════════════════════════════════════════════════════════════════════
    def _position(self, X, s, f, ePx, sl0, tp, fill_px, use_trail, be_off=0.0):
        n = X["n"]
        TL = self.params["trailPivLen"]
        use_alt = self.params["useAltBE"]
        if s == 1:
            A_o, A_h, A_l, A_c, A_t = X["o"], X["h"], X["l"], X["c"], X["hl"]
            up_ptr, up_lv = X["ptrH"], X["lvH"]
            dn_ptr, dn_lv, dn_pv = X["ptrL"], X["lvL"], X["pvL"]
        else:
            A_o, A_h, A_l, A_c, A_t = X["o"], X["l"], X["h"], X["c"], X["lh"]
            up_ptr, up_lv = X["ptrL"], X["lvL"]
            dn_ptr, dn_lv, dn_pv = X["ptrH"], X["lvH"], X["pvH"]
        e, SL, TP, fp = ePx * s, sl0 * s, tp * s, fill_px * s
        be, ext, profT, alt = False, e, -1, None

        be_lvl = e - be_off  # BE raven (be_off točk v minusu, da je ne pobere spread)

        def why(sl):
            return "trail" if sl > e else ("be" if be and sl >= be_lvl - 1e-9 else "sl")

        W = 256
        k = f
        a0 = f
        while a0 < n:
            b0 = min(n, a0 + W)
            O = (A_o[a0:b0] * s).tolist()
            H = (A_h[a0:b0] * s).tolist()
            Lw = (A_l[a0:b0] * s).tolist()
            C = (A_c[a0:b0] * s).tolist()
            TR = (A_t[a0:b0] * s).tolist()
            for k in range(a0, b0):
                i = k - a0
                ok, hk, lk = O[i], H[i], Lw[i]
                if k == f:
                    gap = ok <= e                  # izvršen po open (open že čez nivo)
                    if lk <= SL:
                        return k, (ok if (gap and ok <= SL) else SL) * s, why(SL)
                    if gap and hk >= TP:
                        return k, TP * s, "tp"
                else:
                    if ok <= SL:
                        return k, ok * s, why(SL)
                    if ok >= TP:
                        return k, ok * s, "tp"
                    if lk <= SL:                   # SL ima prednost, če sta oba v isti sveči
                        return k, SL * s, why(SL)
                    if hk >= TP:
                        return k, TP * s, "tp"
                # ── upravljanje ob zaprtju sveče k ──
                if not be:
                    hit = False
                    for qi in range(up_ptr[k], up_ptr[k + 1]):
                        v = up_lv[qi] * s
                        if e < v < TP:
                            hit = True
                            break
                    if not hit and use_alt and alt is not None and hk >= alt:
                        hit = True
                    if hit:
                        be = True
                        if be_lvl > SL:
                            SL = be_lvl
                    elif use_alt and alt is None and profT >= 0:
                        for qi in range(dn_ptr[k], dn_ptr[k + 1]):
                            if dn_lv[qi] * s < e and dn_pv[qi] >= f and profT > dn_pv[qi]:
                                alt = ext
                                break
                if k > f:
                    if hk > ext:
                        ext = hk
                    if hk > e:
                        profT = k
                    if use_trail:
                        t = TR[i]
                        if t == t and (k - TL) > f and t > SL and t < C[i]:
                            SL = t
                            if SL >= e:
                                be = True
            a0 = b0
            W = min(W * 4, 1 << 16)
        return n - 1, X["c"][n - 1], "end"

    # ══════════════════════════════════════════════════════════════════════
    # SIMULATE (na kombinacijo add-onov)
    # ══════════════════════════════════════════════════════════════════════
    def simulate_trades(self, df: pd.DataFrame, addons: dict, ctx) -> pd.DataFrame:
        X = ctx
        n, cd = X["n"], X["cand"]
        cols = ["entry_time", "exit_time", "dir", "entry", "exit", "sl", "tp", "reason"]
        if X["m"] == 0:
            return pd.DataFrame(columns=cols)
        asia, nyf, nwg = addons.get("asia_sweep_filter"), addons.get("ny_sweep_filter"), addons.get("nwog_filter")
        v = int(bool(asia)) | (int(bool(nyf)) << 1) | (int(bool(nwg)) << 2)

        # f_dirAllowed na vsako svečo
        blockL = np.zeros(n, bool)
        blockS = np.zeros(n, bool)
        if asia:
            blockL |= X["asb"] == -1
            blockS |= X["asb"] == 1
        if nyf:
            blockL |= X["in_ny"] & (X["nyb"] == -1)
            blockS |= X["in_ny"] & (X["nyb"] == 1)
        if nwg:
            blockS |= X["nwp"]
        allowL, allowS = ~blockL, ~blockS

        # izbor RB-jev
        r, bl = cd["r"], cd["bull"]
        acc = cd["swIn"] if addons.get("sweep_inside_rb") else cd["swAny"]
        if addons.get("fvg_delivery"):
            acc = acc & cd["deliv"]
        if addons.get("star_hammer_filter"):
            acc = acc & ~cd["star"]
        acc = acc & np.where(bl, allowL[r], allowS[r])
        R = np.minimum(cd["R_common"], cd["R_touch_v"][:, v])
        if addons.get("invalidation"):
            R = np.minimum(R, cd["R_inv"])
        ids = np.flatnonzero(acc & (R > r))

        # najnovejši živ setup na svečo (ločeno long / short), nato filter smeri
        candL = np.full(n, -1, np.int64)
        candS = np.full(n, -1, np.int64)
        for i in ids:
            if bl[i]:
                candL[r[i]:R[i]] = i
            else:
                candS[r[i]:R[i]] = i
        useL = (candL >= 0) & allowL
        useS = (candS >= 0) & allowS
        pickS = useS & (~useL | (candS > candL))
        pickL = useL & ~pickS
        chosen = np.where(pickL, candL, np.where(pickS, candS, -1))
        place = (chosen >= 0) & X["in_sess"] & ~X["bo_next"]
        ePx_ext = np.concatenate([cd["ePx"], [np.nan]])
        px = ePx_ext[chosen]
        fill = np.zeros(n, bool)
        fill[:-1] = place[:-1] & ((pickL[:-1] & (X["l"][1:] <= px[:-1])) | (pickS[:-1] & (X["h"][1:] >= px[:-1])))
        fj = np.flatnonzero(fill)

        o, tarr = X["o"], X["time"]
        use_trail = bool(addons.get("trailing_sl"))
        sl_buf = float(addons.get("sl_buffer", 0))   # točke: SL dodatno odmaknjen (MT5 izvedba)
        be_off = float(addons.get("be_offset", 0))   # točke: BE toliko v minusu
        out = []
        free = 0
        while True:
            k = np.searchsorted(fj, free)
            if k >= len(fj):
                break
            j = int(fj[k])
            f = j + 1
            sid = int(chosen[j])
            bull = bool(bl[sid])
            s = 1 if bull else -1
            e = float(cd["ePx"][sid])
            sl0 = float(cd["sl"][sid])
            stop = abs(e - sl0)
            tp, src = self._targets(X, bull, e, stop, j)
            fp = min(o[f], e) if bull else max(o[f], e)
            sl_real = sl0 - s * sl_buf
            x, xp, why = self._position(X, s, f, e, sl_real, tp, fp, use_trail, be_off)
            # lot iz signala (TV): razdalja vstop - originalni SL, brez bufferja in spreada
            out.append((tarr[f], tarr[x], s, fp, float(xp), sl_real, tp, why, stop,
                        tarr[r[sid]], src, j, x))
            free = x
        if not out:
            return pd.DataFrame(columns=cols)
        res = pd.DataFrame(out, columns=cols + ["risk_dist", "rb_time", "tp_src", "order_bar", "exit_bar"])
        return res
