"""Primer strategije: EMA crossover s tremi add-oni. Služi kot predloga in test cevovoda."""
import numpy as np
import pandas as pd

from strategies.base import Strategy, atr, ema, rsi


class EmaCross(Strategy):
    name = "example_ema_cross"
    description = "EMA 20/50 crossover, SL = swing / ATR, TP = 2R."
    addons = {
        "trend_filter": False,    # trade samo v smeri EMA 200
        "session_filter": False,  # trade samo 13:00-20:00 (čas strežnika), NY seja
        "rsi_filter": False,      # long samo RSI > 50, short samo RSI < 50
    }
    params = {"fast": 20, "slow": 50, "trend": 200, "atr_n": 14, "sl_atr": 1.5, "rr": 2.0}

    def generate_signals(self, df: pd.DataFrame, addons: dict[str, bool]) -> pd.DataFrame:
        p = self.params
        fast, slow = ema(df["close"], p["fast"]), ema(df["close"], p["slow"])
        a = atr(df, p["atr_n"])

        long_ = (fast > slow) & (fast.shift() <= slow.shift())
        short = (fast < slow) & (fast.shift() >= slow.shift())

        if addons.get("trend_filter"):
            t = ema(df["close"], p["trend"])
            long_ &= df["close"] > t
            short &= df["close"] < t
        if addons.get("session_filter"):
            h = df["time"].dt.hour
            ok = (h >= 13) & (h < 20)
            long_ &= ok
            short &= ok
        if addons.get("rsi_filter"):
            r = rsi(df["close"])
            long_ &= r > 50
            short &= r < 50

        signal = np.where(long_, 1, np.where(short, -1, 0))
        dist = p["sl_atr"] * a
        sl = np.where(signal == 1, df["close"] - dist, np.where(signal == -1, df["close"] + dist, np.nan))
        tp = np.where(signal == 1, df["close"] + p["rr"] * dist,
                      np.where(signal == -1, df["close"] - p["rr"] * dist, np.nan))
        return pd.DataFrame({"signal": signal, "sl": sl, "tp": tp}, index=df.index)
