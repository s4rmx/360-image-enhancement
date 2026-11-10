"""CLI: build report.md from metrics + votes."""

from __future__ import annotations

import argparse
import logging
import platform
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

ENHANCE_METHODS = ["clahe", "msrcr", "lime", "fusion", "llf"]


def hardware_blob() -> str:
    try:
        import cpuinfo  # type: ignore

        cpu = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except Exception:  # noqa: BLE001
        cpu = platform.processor() or "unknown"
    # /proc/cpuinfo model name
    model = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except Exception:  # noqa: BLE001
        model = cpu
    cores = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory().total / (1024**3)
    return f"- CPU: {model}\n- Cores: {cores}\n- RAM: {ram:.1f} GiB\n- Platform: {platform.platform()}"


def load_votes(db: Path) -> pd.DataFrame:
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db)
    try:
        return pd.read_sql_query("SELECT * FROM votes", con)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    finally:
        con.close()


def vote_stats(votes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (win_rates, pairwise_matrix, both_worse_rate)."""
    methods = sorted(
        set(votes["method_a"]).union(votes["method_b"]) if len(votes) else ENHANCE_METHODS
    )
    wins = {m: 0.0 for m in methods}
    comps = {m: 0 for m in methods}
    both_worse = {m: 0 for m in methods}
    both_worse_n = {m: 0 for m in methods}
    pair = pd.DataFrame(0.0, index=methods, columns=methods)

    for _, r in votes.iterrows():
        a, b, choice = r["method_a"], r["method_b"], r["choice"]
        for m in (a, b):
            comps[m] = comps.get(m, 0) + 1
            both_worse_n[m] = both_worse_n.get(m, 0) + 1
        if choice == "A":
            wins[a] = wins.get(a, 0) + 1
            pair.loc[a, b] += 1
        elif choice == "B":
            wins[b] = wins.get(b, 0) + 1
            pair.loc[b, a] += 1
        elif choice == "Tie":
            wins[a] = wins.get(a, 0) + 0.5
            wins[b] = wins.get(b, 0) + 0.5
            pair.loc[a, b] += 0.5
            pair.loc[b, a] += 0.5
        elif choice == "BothWorse":
            both_worse[a] = both_worse.get(a, 0) + 1
            both_worse[b] = both_worse.get(b, 0) + 1

    win_rate = pd.Series({m: (wins[m] / comps[m] if comps[m] else float("nan")) for m in methods})
    bw_rate = pd.Series(
        {m: (both_worse[m] / both_worse_n[m] if both_worse_n[m] else float("nan")) for m in methods}
    )
    return win_rate.to_frame("win_rate"), pair, bw_rate


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        c
        for c in df.columns
        if c not in {"dataset", "frame_id", "method"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    rows = []
    for method, g in df.groupby("method"):
        row = {"method": method}
        for m in metrics:
            row[f"{m}_median"] = g[m].median()
            q1, q3 = g[m].quantile(0.25), g[m].quantile(0.75)
            row[f"{m}_iqr"] = q3 - q1
        rows.append(row)
    return pd.DataFrame(rows)


def gate_results(df: pd.DataFrame, bw_rate: pd.Series, runtime: pd.DataFrame) -> pd.DataFrame:
    rows = []
    methods = sorted(df["method"].unique())
    for m in methods:
        g = df[df["method"] == m]
        seam_ok = bool((g["seam_delta"].abs() < 1.0).all()) if "seam_delta" in g else False
        crashes = 0
        if not runtime.empty:
            crashes = int(((runtime["method"] == m) & (runtime["status"] == "failed")).sum())
        bw = float(bw_rate.get(m, float("nan")))
        bw_ok = (not np.isnan(bw)) and bw < 0.30
        # If no votes yet, don't fail the bw gate — mark pending
        if np.isnan(bw):
            bw_ok = True
            bw_note = "pending (no votes)"
        else:
            bw_note = f"{bw:.1%}"
        rows.append(
            {
                "method": m,
                "seam_ok": seam_ok,
                "crashes": crashes,
                "both_worse": bw_note,
                "both_worse_ok": bw_ok,
                "pass": seam_ok and crashes == 0 and bw_ok,
            }
        )
    return pd.DataFrame(rows)


def rank_score(df: pd.DataFrame, win_rate: pd.Series, gates: pd.DataFrame) -> pd.DataFrame:
    passed = set(gates.loc[gates["pass"], "method"])
    # Restrict to enhance methods that passed (ignore original/wb_only for ranking)
    cand = [m for m in ENHANCE_METHODS if m in passed and m in set(df["method"])]
    if not cand:
        cand = [m for m in ENHANCE_METHODS if m in set(df["method"])]

    def med(method, col):
        return df.loc[df["method"] == method, col].median()

    # Lower-is-better ranks
    def ranks(values: dict, higher_better: bool = False) -> dict:
        items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_better)
        return {m: i + 1 for i, (m, _) in enumerate(items)}

    niqe = {m: med(m, "niqe") for m in cand}
    loe = {m: med(m, "loe") for m in cand}
    noise = {m: med(m, "noise_amp") for m in cand}
    clip = {m: med(m, "clipping_pct") for m in cand}
    runtime = {m: med(m, "seconds") if "seconds" in df.columns else 0.0 for m in cand}
    votes = {m: float(win_rate.get(m, 0.0)) if m in win_rate.index else 0.0 for m in cand}

    r_niqe = ranks(niqe, higher_better=False)
    r_loe = ranks(loe, higher_better=False)
    r_noise = ranks(noise, higher_better=False)
    r_clip = ranks(clip, higher_better=False)
    r_rt = ranks(runtime, higher_better=False)
    r_vote = ranks(votes, higher_better=True)

    rows = []
    for m in cand:
        natural = 0.5 * (r_niqe[m] + r_loe[m])
        artifacts = 0.5 * (r_noise[m] + r_clip[m])
        score = 0.50 * r_vote[m] + 0.20 * natural + 0.20 * artifacts + 0.10 * r_rt[m]
        rows.append(
            {
                "method": m,
                "score": score,
                "vote_rank": r_vote[m],
                "natural_rank": natural,
                "artifact_rank": artifacts,
                "runtime_rank": r_rt[m],
            }
        )
    out = pd.DataFrame(rows).sort_values("score")
    out["rank"] = range(1, len(out) + 1)
    return out


def per_dataset_notes(df: pd.DataFrame) -> str:
    lines = []
    if "niqe" not in df.columns:
        return ""
    for dataset, g in df.groupby("dataset"):
        # best NIQE among enhance methods
        sub = g[g["method"].isin(ENHANCE_METHODS)]
        if sub.empty:
            continue
        med = sub.groupby("method")["niqe"].median().sort_values()
        lines.append(f"- **{dataset}**: best NIQE = `{med.index[0]}` ({med.iloc[0]:.3f}); "
                     f"worst = `{med.index[-1]}` ({med.iloc[-1]:.3f})")
    return "\n".join(lines)


def build_report(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing {metrics_path}; run enhance.benchmark first")
    df = pd.read_csv(metrics_path)
    runtime = pd.read_csv(run_dir / "runtime.csv") if (run_dir / "runtime.csv").exists() else pd.DataFrame()
    votes = load_votes(run_dir / "votes.sqlite")
    win_df, pair, bw = vote_stats(votes) if len(votes) else (pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))
    win_rate = win_df["win_rate"] if len(win_df) else pd.Series(dtype=float)

    agg = aggregate_metrics(df)
    gates = gate_results(df, bw, runtime)
    ranking = rank_score(df, win_rate, gates)

    cfg_txt = ""
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        cfg_txt = cfg_path.read_text()

    manifest = ROOT / "data" / "manifest.csv"
    ds_summary = ""
    if manifest.exists():
        man = pd.read_csv(manifest)
        ds_summary = man.groupby("dataset").size().to_string()

    lines = [
        "# 360 Image Auto-Enhance — Benchmark Report",
        "",
        f"Run: `{run_dir.name}`",
        "",
        "## Hardware",
        hardware_blob(),
        "",
        "## Dataset summary",
        "```",
        ds_summary,
        "```",
        "",
        "## Per-method aggregates (median / IQR)",
        agg.to_markdown(index=False) if hasattr(agg, "to_markdown") else agg.to_string(index=False),
        "",
        "## Hard gates",
        gates.to_markdown(index=False) if hasattr(gates, "to_markdown") else gates.to_string(index=False),
        "",
        "## Blind-vote results",
    ]
    if len(votes) == 0:
        lines.append("_No votes recorded yet. Open the Streamlit Blind Vote page._")
    else:
        lines.append(win_df.to_markdown() if hasattr(win_df, "to_markdown") else win_df.to_string())
        lines.append("")
        lines.append("Pairwise wins (row beats column):")
        lines.append(pair.to_markdown() if hasattr(pair, "to_markdown") else pair.to_string())
        lines.append("")
        lines.append("Both-worse-than-original rate:")
        lines.append(bw.to_string())
    lines += [
        "",
        "## Final ranking",
        ranking.to_markdown(index=False) if hasattr(ranking, "to_markdown") else ranking.to_string(index=False),
        "",
        "## Per-dataset differences",
        per_dataset_notes(df) or "_n/a_",
        "",
        "## Recommendation",
    ]
    if len(ranking):
        winner = ranking.iloc[0]["method"]
        lines.append(
            f"**Recommended method: `{winner}`** based on the weighted rank score "
            f"(votes 50%, naturalness 20%, artifacts 20%, runtime 10%). "
            f"Revisit if indoor/outdoor winners diverge above."
        )
    else:
        lines.append("_Insufficient data to recommend._")
    lines += ["", "## Frozen config", "```yaml", cfg_txt, "```", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build report.md for a run")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = Path(args.run)
    text = build_report(run_dir)
    out = run_dir / "report.md"
    out.write_text(text)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
