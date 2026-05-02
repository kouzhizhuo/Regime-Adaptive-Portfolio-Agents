import os
from typing import List
import pandas as _pd
import typer
from rich import print
import json

from profinview.workflows.technical import run_technical_workflow
from profinview.backtest.engine import BacktestRunner, BacktestConfig
from datetime import datetime
from pathlib import Path

app = typer.Typer(no_args_is_help=True, add_completion=False)


def discover_symbols(data_dir: str) -> List[str]:
    syms: List[str] = []
    files = [f for f in os.listdir(data_dir)]
    has_json = any(f.lower().endswith(".json") for f in files)
    has_csv = any(f.lower().endswith(".csv") for f in files)
    if has_json:
        for fn in files:
            if not fn.lower().endswith(".json"):
                continue
            name = os.path.splitext(fn)[0]
            if name.endswith("_data"):
                name = name[:-5]
            syms.append(name.upper())
        return sorted(list({s: None for s in syms}.keys()))
    if has_csv:
        # Prefer aggregated TuShare file if present
        agg = next((f for f in files if f.lower().startswith("tushare_daily") and f.lower().endswith(".csv")), None)
        if agg:
            try:
                df = _pd.read_csv(os.path.join(data_dir, agg), usecols=["ts_code"])
                syms = sorted(list({str(s).upper(): None for s in df["ts_code"].dropna().unique()}.keys()))
                return syms
            except Exception:
                pass
        # Otherwise infer from per-symbol CSVs like 000300.SH.csv
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            base = os.path.splitext(fn)[0]
            sym = base.replace("_", ".").upper()
            syms.append(sym)
        return sorted(list({s: None for s in syms}.keys()))
    return []


@app.command()
def main(
    data_dir: str = typer.Option(..., "--data-dir", help="Directory with SP500 JSON files"),
    symbols: str = typer.Option("AAPL,MSFT", help="Comma-separated tickers to analyze"),
    lookback: int = typer.Option(180, help="Lookback days"),
    workflow: str = typer.Option("technical", help="Workflow name"),
    verbose: bool = typer.Option(True, help="Verbose output"),
    backtest: bool = typer.Option(False, help="Run daily backtest and write results"),
    all_symbols: bool = typer.Option(False, "--all", help="Use all symbols found in data_dir"),
    top_n: int = typer.Option(5, help="Top-N picks each day in strategy"),
    test_start: str = typer.Option(None, help="Test start date YYYY-MM-DD"),
    test_end: str = typer.Option(None, help="Test end date YYYY-MM-DD"),
    rebalance_freq: str = typer.Option("W-FRI", help="Rebalance frequency pandas offset (e.g., W-FRI, M)"),
    rl: bool = typer.Option(False, help="Use RL signal agent with training/validation windows"),
    train_start: str = typer.Option("2021-01-01", help="Training start date YYYY-MM-DD"),
    train_end: str = typer.Option("2023-06-30", help="Training end date YYYY-MM-DD"),
    val_start: str = typer.Option("2023-07-01", help="Validation start date YYYY-MM-DD"),
    val_end: str = typer.Option("2023-12-31", help="Validation end date YYYY-MM-DD"),
    optimizer: bool = typer.Option(False, help="Use mean-variance optimizer path (weekly)"),
    consensus: bool = typer.Option(False, help="Use consensus (router+aggregator) as mu for optimizer"),
    events: bool = typer.Option(False, help="Enable event-driven boosts (FinNLP-like)"),
    alphas: bool = typer.Option(False, help="Enable extra alpha features (Qlib-like)"),
    results_dir: str = typer.Option(None, help="Force results output directory (absolute or relative)"),
    router_policy: str = typer.Option("risk", help="Router policy: risk, diversity"),
    aggregator_policy: str = typer.Option("bayes", help="Aggregator: bayes, precision, median"),
    router_top_k: int = typer.Option(3, help="Router top-k experts to mix"),
    router_min_conf: float = typer.Option(0.05, help="Router minimum expert confidence"),
    signal_cost_bps: float = typer.Option(5.0, help="Per-trade cost in basis points for router utility"),
):
    tickers: List[str]
    if all_symbols:
        tickers = discover_symbols(data_dir)
    else:
        tickers = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if backtest:
        cfg = BacktestConfig(
            data_dir=data_dir,
            symbols=tickers,
            lookback_days=lookback,
            top_n=top_n,
            test_start=test_start,
            test_end=test_end,
            rebalance_freq=rebalance_freq,
            rl=rl,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            optimizer=optimizer,
            consensus=consensus,
            router_policy=router_policy,
            aggregator_policy=aggregator_policy,
            router_top_k=router_top_k,
            router_min_conf=router_min_conf,
            signal_cost_bps=signal_cost_bps,
        )
        # pass flags via env (engine reads)
        os.environ['PROFINVIEW_EVENTS'] = '1' if events else '0'
        os.environ['PROFINVIEW_ALPHAS'] = '1' if alphas else '0'

        runner = BacktestRunner(cfg)
        if results_dir:
            rd = Path(results_dir)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sym_slug = ("ALL" if all_symbols else ",".join(tickers))
            if len(sym_slug) > 50:
                sym_slug = sym_slug[:50] + "_more"
            suffix = f"_lb{lookback}_top{top_n}_{rebalance_freq}" + ("_RL" if rl else "") + ("_OPT" if optimizer else "") + ("_CONS" if consensus else "") + ("_EVT" if events else "") + ("_ALPHA" if alphas else "") + (f"_R{router_policy}" if consensus else "") + (f"_A{aggregator_policy}" if consensus else "") + (f"_{test_start}_to_{test_end}" if test_start or test_end else "")
            rd = Path("backtests") / f"{ts}" / f"{sym_slug}{suffix}"
        summary = runner.run(str(rd), make_plots=True)
        if verbose:
            print({"results_dir": str(rd), "summary": summary})
        print(json.dumps({"results_dir": str(rd), "summary": summary}))
        return

    if workflow == "technical":
        results = run_technical_workflow(data_dir=data_dir, symbols=tickers, lookback_days=lookback, verbose=verbose)
        print({"summary": results.get("risk_summary"), "orders": results.get("orders")[:5] if results.get("orders") else []})
    else:
        raise typer.BadParameter(f"Unknown workflow: {workflow}")

if __name__ == "__main__":
    app()
