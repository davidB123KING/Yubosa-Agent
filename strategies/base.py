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

    # ── NAČIN 2: strategija sama vodi trade-e ──────────────────────────────
    # Za strategije z limit vstopi, premikajočim SL (BE, trailing), dinamičnim TP ipd.
    # Če razred prepiše simulate_trades, motor NE kliče generate_signals.

    def prepare(self, df: pd.DataFrame):
        """Neobvezno. Enkrat na par (pred vsemi kombinacijami) izračuna vse, kar ni odvisno
        od add-onov (pivoti, FVG-ji, seje ...). Vrnjeni objekt dobi simulate_trades kot ctx."""
        return None

    def simulate_trades(self, df: pd.DataFrame, addons: dict[str, bool], ctx) -> pd.DataFrame:
        """Vrne en trade na vrstico, vse cene na BID (kot v podatkih), brez lotov/stroškov:
          entry_time, exit_time : pd.Timestamp
          dir                   : 1 long / -1 short
          entry, exit           : cena vstopa / izhoda
          sl                    : ZAČETNI stop loss (iz njega motor izračuna lot: risk_per_trade, max_lot)
          tp                    : (neobvezno) TP ob vstopu
          reason                : 'tp' / 'sl' / 'be' / 'trail' / 'end' ...
        Motor nato doda spread (en spread na trade), provizijo, lote, pnl, R, metrike, IS/OOS, graf.
        Pravila: ena pozicija naenkrat, brez pogleda v prihodnost, če sta SL in TP zadeta v isti
        sveči, šteje SL."""
        raise NotImplementedError

    def uses_own_simulation(self) -> bool:
        return type(self).simulate_trades is not Strategy.simulate_trades


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
