"""SQLite shema in zapis rezultatov (bohak.db)."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategije (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ime TEXT UNIQUE NOT NULL,
    opis TEXT,
    datoteka TEXT,
    addoni TEXT,              -- JSON seznam imen add-onov
    ustvarjeno TEXT
);
CREATE TABLE IF NOT EXISTS testi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategija_id INTEGER NOT NULL REFERENCES strategije(id),
    timeframe TEXT,
    pari TEXT,                -- JSON
    od TEXT,
    do TEXT,
    split_cas TEXT,           -- JSON {par: čas meje IS/OOS}
    config TEXT,              -- JSON (račun, risk, stroški ...)
    zagnano TEXT
);
CREATE TABLE IF NOT EXISTS rezultati (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL REFERENCES testi(id),
    par TEXT NOT NULL,
    kombinacija TEXT NOT NULL, -- JSON {addon: bool}
    vzorec TEXT NOT NULL CHECK (vzorec IN ('in', 'out')),
    st_tradov INTEGER,
    win_rate REAL,
    profit_factor REAL,
    expectancy_r REAL,
    max_dd REAL,              -- v valuti računa
    max_dd_pct REAL,
    net_profit REAL,
    graf_pot TEXT
);
CREATE INDEX IF NOT EXISTS ix_rez_test ON rezultati(test_id);
"""


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_strategy(con, ime: str, opis: str, datoteka: str, addoni: list[str]) -> int:
    con.execute(
        """INSERT INTO strategije (ime, opis, datoteka, addoni, ustvarjeno) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(ime) DO UPDATE SET opis=excluded.opis, datoteka=excluded.datoteka,
           addoni=excluded.addoni""",
        (ime, opis, datoteka, json.dumps(addoni), datetime.now().isoformat(timespec="seconds")),
    )
    return con.execute("SELECT id FROM strategije WHERE ime=?", (ime,)).fetchone()["id"]


def create_test(con, strategija_id: int, timeframe: str, pari: list[str], od: str, do: str,
                split: dict, config: dict) -> int:
    cur = con.execute(
        "INSERT INTO testi (strategija_id, timeframe, pari, od, do, split_cas, config, zagnano) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (strategija_id, timeframe, json.dumps(pari), od, do, json.dumps(split), json.dumps(config),
         datetime.now().isoformat(timespec="seconds")),
    )
    return cur.lastrowid


def insert_result(con, test_id: int, par: str, kombinacija: dict, vzorec: str, m: dict, graf_pot: str | None):
    con.execute(
        """INSERT INTO rezultati (test_id, par, kombinacija, vzorec, st_tradov, win_rate, profit_factor,
           expectancy_r, max_dd, max_dd_pct, net_profit, graf_pot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (test_id, par, json.dumps(kombinacija), vzorec, m["st_tradov"], m["win_rate"], m["profit_factor"],
         m["expectancy_r"], m["max_dd"], m["max_dd_pct"], m["net_profit"], graf_pot),
    )
