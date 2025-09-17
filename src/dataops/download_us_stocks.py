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

# DB_PATH = "../../data/processed/db.sqlite3"
DB_PATH = "db.sqlite3"
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


def chunk(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def resolve_symbols(include_otc: bool, limit_symbols: Optional[int], symbols_arg: Optional[str]) -> List[str]:
    """
    优先使用 --symbols 白名单；否则回退到可交易清单（主板/非OTC）。
    """
    if symbols_arg:
        return [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    return list_tradable_symbols(include_otc, limit_symbols)


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
        "adjustment": "raw",
        "feed": DATA_FEED
    }
    for attempt in range(8):
        try:
            while True:
                cur = dict(params)
                if token:
                    cur["page_token"] = token
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


def fetch_corporate_actions(start: str, end: str, symbols: Optional[List[str]] = None) -> List[Dict]:
    url = f"{DATA_BASE}/v1/corporate-actions"

    TYPES = "forward_split,reverse_split,unit_split,stock_dividend,cash_dividend"

    def request_once(need_types: bool, syms_chunk: Optional[List[str]]):
        params = {"start": start, "end": end, "limit": 1000}
        if need_types:
            params["types"] = TYPES
        if syms_chunk:
            params["symbols"] = ",".join(syms_chunk)
        resp = requests.get(url, headers=HDR, params=params, timeout=30)
        if resp.status_code == 400 and need_types:
            raise RuntimeError(f"CA 400 types: {resp.text}")
        resp.raise_for_status()
        return resp.json()

    def _normalize_corporate_actions_payload(data):
        flat = []
        if isinstance(data, dict) and isinstance(data.get("corporate_actions"), dict):
            root = data["corporate_actions"]
            groups = ("forward_splits", "reverse_splits", "unit_splits", "stock_dividends", "cash_dividends")
            for g in groups:
                for x in root.get(g, []) or []:
                    y = dict(x);
                    y["type"] = g
                    flat.append(y)
        elif isinstance(data, list):
            flat = data
        else:
            flat = data.get("data") or data.get("items") or []

        out = []
        for x in flat:
            t = str(x.get("type", "")).lower()
            sym = x.get("symbol") or x.get("ticker")
            exd = x.get("ex_date") or x.get("exDate")
            if not sym or not exd:
                continue

            # --- 拆分 / 送股（按比率 old/new 处理） ---
            if ("split" in t) or (t == "stock_dividends"):
                old_r = x.get("old_rate") or x.get("oldRate")
                new_r = x.get("new_rate") or x.get("newRate")
                # 有些 stock_dividend 可能不给 old/new，只给 'rate'（比如 0.1 表示 10% 送股）
                if (old_r is None or new_r is None) and ("stock_dividend" in t):
                    try:
                        r = float(x.get("rate"))
                        old_r, new_r = 1.0, 1.0 + r
                    except Exception:
                        old_r = new_r = None
                try:
                    if old_r is not None and new_r is not None:
                        out.append({
                            "symbol": sym, "ex_date": exd, "ca_type": "split",
                            "old_rate": float(old_r), "new_rate": float(new_r),
                            "cash": None,
                            "record_date": x.get("record_date"), "payable_date": x.get("payable_date"),
                            "process_date": x.get("process_date"), "raw_json": str(x)
                        })
                except Exception:
                    pass
                continue

            # --- 现金分红（金额可能叫 amount/cash/rate） ---
            if ("cash_dividend" in t) or ("dividend" in t and "cash" in str(x).lower()):
                amt = x.get("amount")
                if amt is None: amt = x.get("cash")
                if amt is None: amt = x.get("rate")
                try:
                    amt = float(amt) if amt is not None else None
                except Exception:
                    amt = None
                if amt is not None:
                    out.append({
                        "symbol": sym, "ex_date": exd, "ca_type": "dividend",
                        "old_rate": None, "new_rate": None, "cash": amt,
                        "record_date": x.get("record_date"), "payable_date": x.get("payable_date"),
                        "process_date": x.get("process_date"), "raw_json": str(x)
                    })
                continue
        return out

    rows: List[Dict] = []
    sym_chunks = [None] if not symbols else chunk(symbols, 200)
    for syms in sym_chunks:
        page_token = None
        need_types = True  # 先带 types；遇到 400 再回退
        while True:
            try:
                data = request_once(need_types, syms)
            except RuntimeError as e:  # 400：types 不被接受
                need_types = False
                data = request_once(need_types, syms)
            rows.extend(_normalize_corporate_actions_payload(data))
            # 分页（若存在）
            if isinstance(data, dict):
                page_token = data.get("next_page_token")
                if page_token:
                    # 下一页再拉：注意附加 page_token
                    # 为简洁起见，直接重调 request（Alpaca 大多是基于 token 的）
                    # 这里也可把 page_token 放在 params 里继续循环
                    params = {"start": start, "end": end, "limit": 1000, "page_token": page_token}
                    if need_types: params["types"] = TYPES
                    if syms: params["symbols"] = ",".join(syms)
                    resp = requests.get(url, headers=HDR, params=params, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    rows.extend(_normalize_corporate_actions_payload(data))
                    page_token = data.get("next_page_token")
                    if page_token:
                        continue
            break

    seen = set()
    uniq = []
    for r in rows:
        key = (r["symbol"], r["ex_date"], r["ca_type"])
        if key not in seen:
            seen.add(key);
            uniq.append(r)
    return uniq


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


def init_full(start: str, end: Optional[str], include_otc: bool, limit_symbols: Optional[int], batch: int,
              symbols_arg: Optional[str] = None):
    ensure_schema()
    end = end or utc_today()
    syms = resolve_symbols(include_otc, limit_symbols, symbols_arg)
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


def daily_update(days_back: int, include_otc: bool, limit_symbols: Optional[int], batch: int,
                 symbols_arg: Optional[str] = None):
    ensure_schema()
    end_d = datetime.now(timezone.utc).date()
    start_d = end_d - timedelta(days=days_back)
    s, e = start_d.strftime("%Y-%m-%d"), end_d.strftime("%Y-%m-%d")
    syms = resolve_symbols(include_otc, limit_symbols, symbols_arg)
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
    p_init.add_argument("--symbols", help="逗号分隔的标的清单，如 AAPL,TSLA,SPY")

    p_upd = sub.add_parser("update", help="补最近N天 + 更新公司行动 + 重算复权")
    p_upd.add_argument("--days", type=int, default=3)
    p_upd.add_argument("--include-otc", action="store_true")
    p_upd.add_argument("--limit-symbols", type=int, default=None)
    p_upd.add_argument("--batch", type=int, default=50)
    p_upd.add_argument("--symbols", help="逗号分隔的标的清单，如 AAPL,TSLA,SPY")

    p_adj = sub.add_parser("adjust", help="仅重算复权列")
    p_adj.add_argument("--symbol", default=None)

    p_ca = sub.add_parser("ca", help="仅回填公司行动（可选立即重算）")
    p_ca.add_argument("--symbols", required=True, help="逗号分隔，如 AAPL,TSLA,SPY")
    p_ca.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_ca.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_ca.add_argument("--adjust", action="store_true", help="回填后立即对这些票重算复权")

    args = ap.parse_args()
    if args.cmd == "init":
        init_full(args.start, args.end, args.include_otc, args.limit_symbols, args.batch, symbols_arg=args.symbols)
    elif args.cmd == "update":
        daily_update(args.days, args.include_otc, args.limit_symbols, args.batch, symbols_arg=args.symbols)
    elif args.cmd == "adjust":
        adjust(args.symbol)
    elif args.cmd == "ca":
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        ensure_schema()
        print(f"[ca] backfill CA for {len(syms)} symbols, {args.start}..{args.end}")
        cas = fetch_corporate_actions(args.start, args.end, syms)
        upsert_corporate_actions(cas)
        print(f"[ca] upserted CA rows: {len(cas)}")
        if args.adjust:
            for s in syms:
                recompute_adjusted_for_symbol(s)
            print(f"[ca] adjusted {len(syms)} symbols")


if __name__ == "__main__":
    main()
