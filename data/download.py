"""Prenos zgodovinskih OHLC podatkov iz MetaTrader 5 v lokalne Parquet datoteke.

Primer:
    python data/download.py --pairs US30.cash,EURUSD --tf M5 --years 3

MT5 terminal mora biti zagnan in prijavljen. Obstoječe datoteke se dopolnijo
samo z manjkajočimi bari (--force za ponoven prenos vsega).
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("Manjka paket MetaTrader5. Zaženi: .venv\\Scripts\\pip install -r requirements.txt")

DATA_DIR = Path(__file__).resolve().parent

TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}
CHUNK_DAYS = 90  # prenos po kosih, da obidemo omejitev "Max bars in chart"


def save_meta(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    meta = {
        "symbol": symbol,
        "point": info.point,
        "digits": info.digits,
        "tick_size": info.trade_tick_size,
        "tick_value": info.trade_tick_value,  # v valuti računa
        "contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_step": info.volume_step,
        "volume_max": info.volume_max,
        "currency_profit": info.currency_profit,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (DATA_DIR / f"{symbol}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def fetch(symbol: str, tf: int, start: datetime, end: datetime) -> pd.DataFrame:
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        rates = mt5.copy_rates_range(symbol, tf, cur, nxt)
        if rates is not None and len(rates):
            frames.append(pd.DataFrame(rates))
        cur = nxt
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def download(symbol: str, tf_name: str, years: float, force: bool) -> None:
    if not mt5.symbol_select(symbol, True):
        similar = mt5.symbols_get(f"*{symbol.split('.')[0][:4]}*") or []
        names = ", ".join(s.name for s in similar[:15]) or "/"
        print(f"[NAPAKA] Simbol '{symbol}' ne obstaja. Podobni: {names}")
        return

    out = DATA_DIR / f"{symbol}_{tf_name}.parquet"
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=int(365 * years) + 1)

    old = None
    if out.exists() and not force:
        old = pd.read_parquet(out)
        if len(old):
            start = old["time"].max().to_pydatetime().replace(tzinfo=timezone.utc)

    new = fetch(symbol, TIMEFRAMES[tf_name], start, end)
    df = new if old is None else pd.concat([old, new]).drop_duplicates("time", keep="last")
    df = df.sort_values("time").reset_index(drop=True)
    if df.empty:
        print(f"[NAPAKA] {symbol}: MT5 ni vrnil podatkov ({mt5.last_error()}).")
        return

    df.to_parquet(out, index=False)
    save_meta(symbol)
    print(f"[OK] {symbol} {tf_name}: {len(df):,} barov ({df['time'].min()} -> {df['time'].max()}), "
          f"novih {len(new):,} -> {out.name}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Prenos OHLC iz MT5 v data/*.parquet")
    ap.add_argument("--pairs", required=True, help="Seznam simbolov, ločen z vejico (npr. US30.cash,EURUSD)")
    ap.add_argument("--tf", default="M5", choices=TIMEFRAMES.keys())
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--force", action="store_true", help="Ponovno prenesi vse (prepiši obstoječe)")
    args = ap.parse_args()

    if not mt5.initialize():
        sys.exit(f"[NAPAKA] MT5 se ni zagnal: {mt5.last_error()}. Odpri FTMO MetaTrader 5 in se prijavi.")
    try:
        for sym in [p.strip() for p in args.pairs.split(",") if p.strip()]:
            download(sym, args.tf, args.years, args.force)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
