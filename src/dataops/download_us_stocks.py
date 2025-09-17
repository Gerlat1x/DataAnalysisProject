import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

DB_PATH = "../../data/processed/db.sqlite3"
ALLOWED_EXCH = {"NYSE", "NASDAQ", "ARCA", "AMEX", "BATS", "IEX"}

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SEC = os.getenv("APCA_API_SECRET_KEY")
TRADING_BASE = os.getenv("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets")
DATA_BASE = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
DATA_FEED = os.getenv("ALPACA_DATA_FEED", "iex")

if not API_KEY or not API_SEC:
    raise SystemExit("缺少 key/secret：请在 .env 或环境变量中设置 APCA_API_KEY_ID / APCA_API_SECRET_KEY（或 ALPACA_*）。")

HDR = {"Apca-Api-Key-Id": API_KEY, "Apca-Api-Secret-Key": API_SEC, "Accept": "application/json"}


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# ---------- schema ----------
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1d(
  symbol TEXT NOT NULL,
  ts TEXT NOT NULL,          -- ISO8601 UTC
  open REAL NOT NULL,
  high REAL NOT NULL,
  low  REAL NOT NULL,
  close REAL NOT NULL,
  volume INTEGER NOT NULL,
  vwap REAL,
  -- 复权列（可能后续填充）
  adj_open_split REAL, adj_high_split REAL, adj_low_split REAL, adj_close_split REAL,
  adj_open_tr REAL,    adj_high_tr REAL,    adj_low_tr REAL,    adj_close_tr REAL,
  PRIMARY KEY(symbol, ts)
);
CREATE INDEX IF NOT EXISTS idx_b1d_sym_ts ON bars_1d(symbol, ts);

CREATE TABLE IF NOT EXISTS corporate_actions(
  symbol TEXT NOT NULL,
  ex_date TEXT NOT NULL,      -- YYYY-MM-DD
  ca_type TEXT NOT NULL,      -- split | dividend | merger | spin_off | ...
  old_rate REAL,              -- 拆并前份额
  new_rate REAL,              -- 拆并后份额（ratio = new/old）
  cash REAL,                  -- 现金分红/每股金额（若有）
  record_date TEXT,
  payable_date TEXT,
  process_date TEXT,
  raw_json TEXT,              -- 原始记录，便于追溯
  PRIMARY KEY(symbol, ex_date, ca_type)
);
"""


def ensure_schema():
    with get_conn() as c:
        c.executescript(SCHEMA)


# ---------- utils ----------
def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


# ---------- assets ----------
def list_tradable_symbols(include_otc: bool, limit_symbols: Optional[int]) -> List[str]:
    url = f"{TRADING_BASE}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    out, page_token = [], None
    while True:
        p = dict(params)
        if page_token: p["page_token"] = page_token
        r = requests.get(url, headers=HDR, params=p, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("assets", data) if isinstance(data, dict) else data
        for a in items:
            if not a.get("tradable"): continue
            exch = a.get("exchange")
            if not include_otc and exch == "OTC": continue
            if not include_otc and exch not in ALLOWED_EXCH: continue
            out.append(a["symbol"])
            if limit_symbols and len(out) >= limit_symbols:
                return sorted(set(out))
        page_token = data.get("next_page_token") if isinstance(data, dict) else None
        if not page_token: break
    return sorted(set(out))


# ---------- bars ----------
def fetch_bars_daily_batch(symbols: List[str], start_date: str, end_date: str) -> List[Dict]:
    if not symbols: return []
    rows, token = [], None
    url = f"{DATA_BASE}/v2/stocks/bars"
    params = {
        "timeframe": "1Day",
        "symbols": ",".join(symbols),
        "start": f"{start_date}T00:00:00Z",
        "end": f"{end_date}T23:59:59Z",
        "limit": 10000,
        "adjustment": "raw",  # 我们自行复权
        "feed": DATA_FEED
    }
    for attempt in range(8):
        try:
            while True:
                cur = dict(params)
                if token: cur["page_token"] = token
                resp = requests.get(url, headers=HDR, params=cur, timeout=30)
                if resp.status_code == 429:
                    time.sleep(min(60, 0.5 * (2 ** attempt)))
                    continue
                resp.raise_for_status()
                data = resp.json()
                bars_by_sym = data.get("bars", {})
                for sym, bars in bars_by_sym.items():
                    for b in bars:
                        rows.append({
                            "symbol": sym,
                            "ts": b["t"],
                            "open": b["o"],
                            "high": b["h"],
                            "low": b["l"],
                            "close": b["c"],
                            "volume": b["v"],
                            "vwap": b.get("vw")
                        })
                token = data.get("next_page_token")
                if not token: return rows
        except requests.RequestException:
            if attempt == 7: raise
            time.sleep(min(60, 0.5 * (2 ** attempt)))
    return rows


def upsert_bars(rows: List[Dict]):
    if not rows: return
    with get_conn() as conn:
        conn.executemany("""
        INSERT INTO bars_1d(symbol, ts, open, high, low, close, volume, vwap)
        VALUES (:symbol, :ts, :open, :high, :low, :close, :volume, :vwap)
        ON CONFLICT(symbol, ts) DO UPDATE SET
          open=excluded.open, high=excluded.high, low=excluded.low,
          close=excluded.close, volume=excluded.volume, vwap=excluded.vwap
        """, rows)
        conn.commit()


# ---------- corporate actions ----------
def fetch_corporate_actions(start: str, end: str, symbols: Optional[List[str]] = None) -> List[Dict]:
    """
    新接口：/v1/corporate-actions
    允许按 types / start / end / symbols 过滤（官方称支持多过滤）。
    返回统一结构的 list：{symbol, ex_date, ca_type, old_rate, new_rate, cash, ...}
    """
    url = f"{DATA_BASE}/v1/corporate-actions"
    types = "forward_split,reverse_split,cash_dividend"
    rows: List[Dict] = []
    # 尝试批量 symbols 过滤，分块以控制 URL 长度
    sym_chunks = [None] if not symbols else chunk(symbols, 200)

    for syms in sym_chunks:
        token = None
        while True:
            params = {
                "types": types,
                "start": start,
                "end": end,
                "limit": 1000,
            }
            if syms:
                params["symbols"] = ",".join(syms)
            if token:
                params["page_token"] = token

            resp = requests.get(url, headers=HDR, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            # 兼容不同结构：有的返回 {'corporate_actions': {...}}，有的直接平铺
            cas_root = data.get("corporate_actions", data)

            # 可能按类型分组
            def _items(name):
                v = cas_root.get(name, [])
                return v if isinstance(v, list) else []

            forward_splits = _items("forward_splits")
            reverse_splits = _items("reverse_splits")
            cash_dividends = _items("cash_dividends") or _items("dividends")

            def norm_split(x, kind):
                return {
                    "symbol": x.get("symbol"),
                    "ex_date": x.get("ex_date"),
                    "ca_type": "split",
                    "old_rate": float(x.get("old_rate", 1) or 1),
                    "new_rate": float(x.get("new_rate", 1) or 1),
                    "cash": None,
                    "record_date": x.get("record_date"),
                    "payable_date": x.get("payable_date"),
                    "process_date": x.get("process_date"),
                    "raw_json": str(x),
                    "_kind": kind
                }

            def norm_div(x):
                amt = x.get("amount") or x.get("cash")
                try:
                    amt = float(amt) if amt is not None else None
                except:
                    amt = None
                return {
                    "symbol": x.get("symbol"),
                    "ex_date": x.get("ex_date"),
                    "ca_type": "dividend",
                    "old_rate": None,
                    "new_rate": None,
                    "cash": amt,
                    "record_date": x.get("record_date"),
                    "payable_date": x.get("payable_date"),
                    "process_date": x.get("process_date"),
                    "raw_json": str(x),
                    "_kind": "cash_dividend"
                }

            for x in forward_splits: rows.append(norm_split(x, "forward"))
            for x in reverse_splits: rows.append(norm_split(x, "reverse"))
            for x in cash_dividends: rows.append(norm_div(x))

            token = data.get("next_page_token")
            if not token: break

    # 清洗：去掉缺 symbol/ex_date 的项
    rows = [r for r in rows if r["symbol"] and r["ex_date"]]
    return rows


def upsert_corporate_actions(rows: List[Dict]):
    if not rows: return
    with get_conn() as conn:
        conn.executemany("""
        INSERT INTO corporate_actions(symbol, ex_date, ca_type, old_rate, new_rate, cash,
                                      record_date, payable_date, process_date, raw_json)
        VALUES (:symbol, :ex_date, :ca_type, :old_rate, :new_rate, :cash,
                :record_date, :payable_date, :process_date, :raw_json)
        ON CONFLICT(symbol, ex_date, ca_type) DO UPDATE SET
          old_rate=COALESCE(excluded.old_rate, old_rate),
          new_rate=COALESCE(excluded.new_rate, new_rate),
          cash=COALESCE(excluded.cash, cash),
          record_date=COALESCE(excluded.record_date, record_date),
          payable_date=COALESCE(excluded.payable_date, payable_date),
          process_date=COALESCE(excluded.process_date, process_date),
          raw_json=excluded.raw_json
        """, rows)
        conn.commit()


# ---------- adjustment ----------
def _build_split_factors(events: List[Dict]) -> List[Tuple[str, float]]:
    """返回 (ex_date, factor_before) 列表。factor_before 作用于 ex_date 之前的所有历史：
       2拆1(new=2,old=1) => 历史乘 (old/new)=0.5； 1并10(new=1,old=10) => 历史乘 (old/new)=10
    """
    out = []
    for e in events:
        old_r = e.get("old_rate") or 1.0
        new_r = e.get("new_rate") or 1.0
        try:
            f = float(old_r) / float(new_r)
        except:
            continue
        out.append((e["ex_date"], f))
    # 同日多次时合并
    from collections import defaultdict
    dd = defaultdict(float)
    for d, f in out:
        dd[d] = dd[d] + (f - 1) + 1 if dd[d] else f  # 当天连乘
    return sorted(dd.items(), key=lambda x: x[0])


def recompute_adjusted_for_symbol(symbol: str):
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT ts, open, high, low, close FROM bars_1d WHERE symbol=? ORDER BY ts ASC", (symbol,))
        bars = c.fetchall()
        if not bars: return
        # 拉公司行动
        c.execute("SELECT ex_date, ca_type, old_rate, new_rate, cash FROM corporate_actions WHERE symbol=?", (symbol,))
        ca = c.fetchall()

    # 整理
    splits = []
    dividends = []
    for ex_date, ca_type, old_r, new_r, cash in ca:
        if ca_type == "split":
            splits.append({"ex_date": ex_date, "old_rate": old_r, "new_rate": new_r})
        elif ca_type == "dividend":
            if cash is not None: dividends.append({"ex_date": ex_date, "cash": float(cash)})

    # 建映射：date->close
    def to_date(ts_iso: str) -> str:
        return ts_iso[:10]

    dates = [to_date(ts) for (ts, *_rest) in bars]
    closes = [row[4] for row in bars]
    date_to_close = {d: v for d, v in zip(dates, closes)}

    # 预备拆并累计因子：factor(d) = ∏_{ex>s d} f(ex)
    split_changes = _build_split_factors(splits)  # [(ex_date, f_before)]

    # 先把所有 ex_date 的前一日因子落入“分段常数”里
    # 简化：对每一条bar独立计算：查所有 ex_date > d 的乘积（日线规模可接受）
    def split_factor_for_date(d: str) -> float:
        f = 1.0
        for ex, fb in split_changes:
            if ex > d:
                f *= fb
        return f

    # 现金分红的“总回报”因子（用 ex-1 日收盘作为基准）
    # factor_div(d) = ∏_{ex > d} (1 - cash / close(ex-1))
    # 找到 ex 之前最近一个有bar的日期
    date_list = dates  # 升序
    date_index = {d: i for i, d in enumerate(date_list)}

    def prev_trade_date(ex: str) -> Optional[str]:
        # 找 ex 当天的 index；若无，则找到 < ex 的最大 d
        # 简化：二分替代，这里线性回退也可
        for i in range(len(date_list) - 1, -1, -1):
            if date_list[i] < ex:
                return date_list[i]
        return None

    div_factors_cache: Dict[str, float] = {}

    def div_factor_for_date(d: str) -> float:
        if d in div_factors_cache: return div_factors_cache[d]
        f = 1.0
        for e in dividends:
            ex = e["ex_date"]
            if ex > d:
                pday = prev_trade_date(ex)
                if pday and pday in date_to_close and date_to_close[pday] > 0:
                    f *= max(1e-9, 1.0 - float(e["cash"]) / float(date_to_close[pday]))
        div_factors_cache[d] = f
        return f

    # 计算并写回
    updates = []
    for (ts, o, h, l, cl) in bars:
        d = to_date(ts)
        fs = split_factor_for_date(d)
        ft = fs * div_factor_for_date(d)

        adj_split = (o * fs, h * fs, l * fs, cl * fs)
        adj_tr = (o * ft, h * ft, l * ft, cl * ft)

        updates.append((adj_split[0], adj_split[1], adj_split[2], adj_split[3],
                        adj_tr[0], adj_tr[1], adj_tr[2], adj_tr[3], symbol, ts))

    with get_conn() as conn:
        conn.executemany("""
        UPDATE bars_1d
        SET adj_open_split=?, adj_high_split=?, adj_low_split=?, adj_close_split=?,
            adj_open_tr=?,    adj_high_tr=?,    adj_low_tr=?,    adj_close_tr=?
        WHERE symbol=? AND ts=?
        """, updates)
        conn.commit()


# ---------- high-level flows ----------
def init_full(start: str, end: Optional[str], include_otc: bool, limit_symbols: Optional[int], batch: int):
    ensure_schema()
    end = end or utc_today()
    syms = list_tradable_symbols(include_otc, limit_symbols)
    print(f"[init] symbols={len(syms)} range={start}..{end} feed={DATA_FEED} otc={include_otc}")

    # 1) bars
    for i, group in enumerate(chunk(syms, batch), 1):
        rows = fetch_bars_daily_batch(group, start, end)
        upsert_bars(rows)
        print(f"  bars [{i}/{(len(syms) + batch - 1) // batch}] +{len(rows)} rows for {len(group)} syms")

    # 2) corporate actions
    print("  fetching corporate actions...")
    cas = fetch_corporate_actions(start, end, syms)
    upsert_corporate_actions(cas)
    print(f"  upserted CA rows: {len(cas)}")

    # 3) adjust
    print("  recomputing adjusted columns (split + total return)...")
    for k, group in enumerate(chunk(syms, 50), 1):
        for s in group:
            recompute_adjusted_for_symbol(s)
        print(f"    adjust [{k}/{(len(syms) + 49) // 50}]")


def daily_update(days_back: int, include_otc: bool, limit_symbols: Optional[int], batch: int):
    ensure_schema()
    end_d = datetime.now(timezone.utc).date()
    start_d = end_d - timedelta(days=days_back)
    s, e = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    syms = list_tradable_symbols(include_otc, limit_symbols)
    print(f"[update] symbols={len(syms)} range={s}..{e} feed={DATA_FEED} otc={include_otc}")

    # bars
    for i, group in enumerate(chunk(syms, batch), 1):
        rows = fetch_bars_daily_batch(group, s, e)
        upsert_bars(rows)
        print(f"  bars [{i}/{(len(syms) + batch - 1) // batch}] +{len(rows)} rows")

    # CA
    print("  fetching corporate actions...")
    cas = fetch_corporate_actions(s, e, syms)
    upsert_corporate_actions(cas)
    print(f"  upserted CA rows: {len(cas)}")

    # adjust（只对受影响的票重算更快，但先简化成全量重算）
    print("  recomputing adjusted columns...")
    for k, group in enumerate(chunk(syms, 50), 1):
        for s in group:
            recompute_adjusted_for_symbol(s)
        print(f"    adjust [{k}/{(len(syms) + 49) // 50}]")


def adjust(symbol: Optional[str]):
    ensure_schema()
    if symbol:
        recompute_adjusted_for_symbol(symbol.upper())
        print(f"adjusted {symbol}")
        return
    # 全量
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT symbol FROM bars_1d")
        syms = [r[0] for r in c.fetchall()]
    for k, group in enumerate(chunk(syms, 50), 1):
        for s in group:
            recompute_adjusted_for_symbol(s)
        print(f"  adjust [{k}/{(len(syms) + 49) // 50}]")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="全量回填日线 + 公司行动 + 复权")
    p_init.add_argument("--start", required=True)
    p_init.add_argument("--end", default=None)
    p_init.add_argument("--include-otc", action="store_true")
    p_init.add_argument("--limit-symbols", type=int, default=None, help="只取前N个标的做测试")
    p_init.add_argument("--batch", type=int, default=50, help="每次请求最多N个symbol")

    p_upd = sub.add_parser("update", help="补最近N天 + 更新公司行动 + 重算复权")
    p_upd.add_argument("--days", type=int, default=3)
    p_upd.add_argument("--include-otc", action="store_true")
    p_upd.add_argument("--limit-symbols", type=int, default=None)
    p_upd.add_argument("--batch", type=int, default=50)

    p_adj = sub.add_parser("adjust", help="仅重算复权列")
    p_adj.add_argument("--symbol", default=None)

    args = ap.parse_args()
    if args.cmd == "init":
        init_full(args.start, args.end, args.include_otc, args.limit_symbols, args.batch)
    elif args.cmd == "update":
        daily_update(args.days, args.include_otc, args.limit_symbols, args.batch)
    elif args.cmd == "adjust":
        adjust(args.symbol)


if __name__ == "__main__":
    main()
