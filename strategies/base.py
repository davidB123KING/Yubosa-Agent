"""Vmesnik za strategije. Vsaka strategija je svoja datoteka v strategies/
z enim razredom, ki deduje od Strategy.

Motor (engine/) sam poskrbi za izvedbo tradov, stroške, velikost pozicije,
kombinacije add-onov in IS/OOS. Strategija samo vrne signale.
"""
import pandas as pd


class Strategy:
    name: str = "base"
    description: str = ""
    # Add-oni: ime -> privzeta vrednost (bool). Motor testira VSE kombinacije True/False.
    addons: dict[str, bool] = {}
    # Fiksni parametri strategije (ne kombinirajo se).
    params: dict = {}

    def generate_signals(self, df: pd.DataFrame, addons: dict[str, bool]) -> pd.DataFrame:
        """Prejme OHLC df (time, open, high, low, close, tick_volume, spread; cene so bid)
        in slovar vklopljenih add-onov.

        Vrne df z enakim indeksom in stolpci:
          signal : 1 = long, -1 = short, 0 = nič (na zaprtju bara; vstop je na open naslednjega)
          sl     : cena stop lossa (obvezno, ko je signal != 0)
          tp     : cena take profita ali NaN (brez TP)
          exit   : (neobvezno) bool, True = zapri odprto pozicijo na open naslednjega bara
        """
        raise NotImplementedError


# Pomožni indikatorji, ki jih strategije lahko uporabijo.
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)
