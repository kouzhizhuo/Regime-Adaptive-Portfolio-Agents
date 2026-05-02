from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import pandas as pd

REQUIRED_COLUMNS = ["date", "close"]

class SP500JsonLoader:
    def __init__(self, data_dir: str) -> None:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data dir not found: {data_dir}")
        self.data_dir = data_dir

    def _parse_nested_daily(self, obj: dict) -> pd.DataFrame:
        price_data = obj.get("price_data")
        if not isinstance(price_data, dict):
            return pd.DataFrame()
        daily = price_data.get("daily_data") or price_data.get("daily") or price_data.get("Time Series (Daily)")
        if isinstance(daily, dict):
            rows = []
            for date_str, row in daily.items():
                if not isinstance(row, dict):
                    continue
                rec = {"date": date_str}
                # normalize to lowercase keys for easier selection later
                for k, v in row.items():
                    rec[k.lower()] = v
                rows.append(rec)
            if rows:
                df = pd.DataFrame(rows)
                return df
        elif isinstance(daily, list) and daily and isinstance(daily[0], dict):
            df = pd.DataFrame(daily)
            if "date" not in df.columns:
                for c in df.columns:
                    if c.lower() in ["dt", "timestamp", "date_time"]:
                        df = df.rename(columns={c: "date"})
                        break
            return df
        return pd.DataFrame()

    def _select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_lower = {c.lower(): c for c in df.columns}
        def pick(*cands):
            for name in cands:
                if name in cols_lower:
                    return cols_lower[name]
            return None
        date_c = pick("date", "dt", "timestamp", "date_time")
        close_c = pick("adjusted_close", "adj_close", "close", "close_price", "c")
        open_c = pick("open", "o", "open_price")
        high_c = pick("high", "h", "high_price")
        low_c = pick("low", "l", "low_price")
        vol_c = pick("volume", "v", "volume_shares")
        if not date_c or not close_c:
            raise ValueError("Missing required columns after normalization")
        out = pd.DataFrame({"date": df[date_c], "close": df[close_c]})
        if open_c:
            out["open"] = df[open_c]
        if high_c:
            out["high"] = df[high_c]
        if low_c:
            out["low"] = df[low_c]
        if vol_c:
            out["volume"] = df[vol_c]
        return out

    def _read_one(self, path: str) -> pd.DataFrame:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, dict):
            df_nested = self._parse_nested_daily(data)
            if not df_nested.empty:
                df = df_nested
            else:
                list_vals = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
                if list_vals:
                    df = pd.DataFrame(list_vals[0])
                else:
                    df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        df = self._select_columns(df)
        for req in REQUIRED_COLUMNS:
            if req not in df.columns:
                raise ValueError(f"Missing required column '{req}' in {path}")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].sort_values("date").reset_index(drop=True)
        return df

    def load(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        symbol_to_df: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            candidates = [
                os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                if f.lower().startswith(sym.lower()) and f.lower().endswith(".json")
            ]
            if not candidates:
                cand = os.path.join(self.data_dir, f"{sym}.json")
                if os.path.exists(cand):
                    candidates = [cand]
            if not candidates:
                continue
            try:
                df = self._read_one(candidates[0])
                # Filter out too short series
                if len(df) >= 60:
                    symbol_to_df[sym.upper()] = df
            except Exception:
                # Skip symbols without usable OHLC data
                continue
        if not symbol_to_df:
            raise FileNotFoundError("No usable JSON files found in data directory")
        return symbol_to_df


class TuShareDailyCSVLoader:
    """
    Loader for TuShare daily price CSVs.

    Supported layouts:
    1) One aggregated CSV that contains many symbols over many dates with columns:
       ts_code, trade_date, open, high, low, close, vol/amount (volume optional).
    2) One CSV per symbol in the directory named like "000300.SH.csv" with TuShare columns.
    """

    REQUIRED_COLS = ["ts_code", "trade_date", "close"]

    def __init__(self, data_dir: str, aggregated_filename: Optional[str] = None) -> None:
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Data dir not found: {data_dir}")
        self.data_dir = data_dir
        self.aggregated_filename = aggregated_filename

    def _read_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                if c in lower:
                    return lower[c]
            return None
        ts_code = pick("ts_code")
        trade_date = pick("trade_date", "date")
        close = pick("close", "adj_close", "close_price")
        if not (ts_code and trade_date and close):
            raise ValueError("Missing required columns in TuShare CSV")
        out = pd.DataFrame({
            "ts_code": df[ts_code].astype(str),
            "date": pd.to_datetime(df[trade_date].astype(str)).dt.date,
            "close": df[close].astype(float),
        })
        for src, dst in [(pick("open"), "open"), (pick("high"), "high"), (pick("low"), "low"), (pick("vol", "volume"), "volume")]:
            if src:
                out[dst] = df[src]
        return out

    def _load_from_aggregated(self, path: str, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        df = self._read_csv(path)
        # Some aggregated files are not full daily panel; ensure duplicates removed and dates parsed
        if "date" in df.columns:
            df = df.dropna(subset=["date"]).copy()
        result: Dict[str, pd.DataFrame] = {}
        codes = list(dict.fromkeys(df["ts_code"].astype(str).str.upper().tolist()))
        wanted = set(s.upper() for s in symbols) if symbols else set(codes)
        # Restrict to requested symbols first for speed
        df = df[df["ts_code"].astype(str).str.upper().isin(wanted)]
        if df.empty:
            raise FileNotFoundError("No requested ts_code present in aggregated TuShare CSV")
        for code, g in df.groupby("ts_code"):
            g = g.sort_values("date").reset_index(drop=True)
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in g.columns]
            if len(g) >= 60:
                result[str(code).upper()] = g[keep]
        if not result:
            raise FileNotFoundError("No usable TuShare data found in aggregated CSV after filtering")
        return result

    def _load_from_symbol_files(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            # support ts_code form like 000300.SH
            candidates = [
                os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                if f.lower() in {f"{sym.lower()}.csv", f"{sym.lower().replace('.', '_')}.csv"}
            ]
            if not candidates:
                continue
            try:
                g = self._read_csv(candidates[0]).sort_values("date").reset_index(drop=True)
                keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in g.columns]
                if len(g) >= 60:
                    result[sym.upper()] = g[keep]
            except Exception:
                continue
        if not result:
            raise FileNotFoundError("No usable TuShare per-symbol CSVs found")
        return result

    def load(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        # Prefer aggregated file if present/explicit
        if self.aggregated_filename:
            agg_path = os.path.join(self.data_dir, self.aggregated_filename)
            if os.path.exists(agg_path):
                try:
                    res = self._load_from_aggregated(agg_path, symbols)
                    # If none of the requested symbols are present, fall back to per-symbol files
                    if res:
                        return res
                except Exception:
                    pass
        # Try auto-detect aggregated file
        for f in os.listdir(self.data_dir):
            if f.lower().startswith("tushare_daily") and f.lower().endswith(".csv"):
                try:
                    res = self._load_from_aggregated(os.path.join(self.data_dir, f), symbols)
                    if res:
                        return res
                except Exception:
                    break
        # Fallback to per-symbol CSVs
        return self._load_from_symbol_files(symbols)


class CNGenericCSVLoader:
    """
    Fallback robust loader for large CN CSVs (e.g., "Ashare dynamic ... .csv").
    Expects at least columns similar to TuShare: ts_code, trade_date (or date), close.
    Uses python CSV engine and skips bad lines.
    """
    def __init__(self, data_dir: str, filename: str) -> None:
        self.data_dir = data_dir
        self.filename = filename

    def load(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        path = os.path.join(self.data_dir, self.filename)
        # Robust read
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")
        lower = {c.lower(): c for c in df.columns}
        # allow simple chinese column aliases
        alias = {
            "代码": "ts_code",
            "股票代码": "ts_code",
            "名称": "name",
            "交易日": "trade_date",
            "日期": "trade_date",
            "收盘": "close",
            "收盘价": "close",
            "最新价": "close",
            "今开": "open",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
        }
        lower.update({k.lower(): v for k, v in alias.items() if k.lower() not in lower})

        def pick(*cands: str) -> Optional[str]:
            for c in cands:
                key = c.lower()
                if key in lower:
                    mapped = lower[key]
                    if mapped in df.columns:
                        return mapped
                    # alias may map to canonical name; find original column by reverse lookup
                    for orig, val in alias.items():
                        if val == lower[key] and orig in df.columns:
                            return orig
            return None

        ts_code = pick("ts_code", "code", "symbol", "代码", "股票代码")
        trade_date = pick("trade_date", "date", "datetime", "dt", "交易日", "日期")
        close = pick("close", "adj_close", "close_price", "price", "收盘", "收盘价", "最新价")
        if not (ts_code and trade_date and close):
            raise ValueError("Missing required columns in CN generic CSV")
        raw_code = df[ts_code].astype(str).str.strip()
        # Normalize numeric symbol columns like 300472 -> "300472"
        raw_code = raw_code.str.replace(r"\.0$", "", regex=True)
        out = pd.DataFrame({
            "ts_code": raw_code,
            "date": pd.to_datetime(df[trade_date].astype(str)).dt.date,
            "close": pd.to_numeric(df[close], errors="coerce")
        })
        # optional OHLCV
        for src, dst in [(pick("open"), "open"), (pick("high"), "high"), (pick("low"), "low"), (pick("vol", "volume"), "volume")]:
            if src:
                out[dst] = pd.to_numeric(df[src], errors="coerce")
        out = out.dropna(subset=["date", "close"])
        result: Dict[str, pd.DataFrame] = {}
        wanted = set(s.upper() for s in symbols) if symbols else set(out["ts_code"].unique())
        sub = out[out["ts_code"].astype(str).str.upper().isin(wanted)]
        if sub.empty:
            raise FileNotFoundError("No requested ts_code present in CN generic CSV")
        for code, g in sub.groupby("ts_code"):
            g = g.sort_values("date").reset_index(drop=True)
            keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in g.columns]
            if len(g) >= 60:
                result[str(code).upper()] = g[keep]
        if not result:
            raise FileNotFoundError("No usable CN series (>=60 rows) found in generic CSV")
        return result


def get_loader_for_dir(data_dir: str):
    """Auto-detect appropriate loader based on files in the directory."""
    files = [f.lower() for f in os.listdir(data_dir)]
    # Prefer TuShare aggregated daily file if present
    agg = next((f for f in files if f.startswith("tushare_daily") and f.endswith(".csv")), None)
    if agg:
        return TuShareDailyCSVLoader(data_dir, aggregated_filename=agg)
    # Support aligned CN merged exports
    cn_aligned = next((f for f in os.listdir(data_dir) if f.lower() == "cn_price_aligned.csv"), None)
    if cn_aligned:
        return CNGenericCSVLoader(data_dir, filename=cn_aligned)
    # Fallback: CN generic dynamic file if present
    dyn = next((f for f in os.listdir(data_dir) if f.lower().startswith("ashare dynamic")), None)
    if dyn:
        return CNGenericCSVLoader(data_dir, filename=dyn)
    if any(f.endswith(".json") for f in files):
        return SP500JsonLoader(data_dir)
    if any(f.endswith(".csv") for f in files):
        return TuShareDailyCSVLoader(data_dir)
    # default to JSON loader to keep prior behavior, will raise error if unusable
    return SP500JsonLoader(data_dir)
