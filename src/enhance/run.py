"""CLI: batch-process all frames × methods, with per-method resource logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import resource
import yaml
from tqdm import tqdm

from enhance.common import apply_shared_pipeline
from enhance.io import ingest, load_image, resize_preview, save_jpeg
from enhance.methods.base import get_method, list_methods

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

RUNTIME_FIELDS = [
    "ts_utc",
    "dataset",
    "frame_id",
    "method",
    "status",
    "width",
    "height",
    "megapixels",
    "wall_s",          # enhancement only
    "io_s",            # load + save + preview
    "total_s",         # wall_s + io_s
    "cpu_user_s",      # enhancement window
    "cpu_sys_s",
    "cpu_total_s",
    "cpu_util",        # cpu_total_s / wall_s (1.0 = one core fully used)
    "rss_before_mb",
    "rss_after_mb",
    "rss_delta_mb",
    "peak_rss_mb",     # sampled peak during enhancement
    "peak_rss_delta_mb",
    "error",
]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def make_run_id(cfg: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{config_hash(cfg)}-{ts}"


def worker_count() -> int:
    cpu = os.cpu_count() or 2
    ram_gb = psutil.virtual_memory().total / (1024**3)
    return max(1, min(cpu, int(ram_gb // 4)))


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _cpu_times() -> tuple[float, float]:
    """(user_s, system_s) for this process since start."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_utime), float(ru.ru_stime)


class RssSampler:
    """Background thread sampling process RSS during a critical section."""

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RssSampler":
        self.peak_mb = _rss_mb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.peak_mb = max(self.peak_mb, _rss_mb())
            except Exception:  # noqa: BLE001
                break


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    """Worker: enhance one (frame, method). Returns a detailed runtime row."""
    method = task["method"]
    out_full = Path(task["out_full"])
    out_prev = Path(task["out_preview"])
    result: dict[str, Any] = {k: None for k in RUNTIME_FIELDS}
    result.update(
        {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": task["dataset"],
            "frame_id": task["frame_id"],
            "method": method,
            "status": "ok",
            "error": "",
        }
    )

    if task.get("skip_existing") and out_full.exists() and out_prev.exists():
        result["status"] = "skipped"
        return result

    try:
        t_io0 = time.perf_counter()
        img = load_image(Path(task["path"]))
        h, w = img.shape[:2]
        result["width"] = w
        result["height"] = h
        result["megapixels"] = round(w * h / 1e6, 3)
        load_s = time.perf_counter() - t_io0

        shared = task["shared"]
        params = task["params"]
        core = get_method(method)

        rss_before = _rss_mb()
        cpu_u0, cpu_s0 = _cpu_times()
        t0 = time.perf_counter()
        with RssSampler(0.05) as sampler:
            if method == "original":
                out = img.copy()
            else:
                out = apply_shared_pipeline(img, core, params, shared)
        wall_s = time.perf_counter() - t0
        cpu_u1, cpu_s1 = _cpu_times()
        rss_after = _rss_mb()

        t_save0 = time.perf_counter()
        save_jpeg(out_full, out, source_path=Path(task["path"]), quality=int(task["jpeg_quality"]))
        prev = resize_preview(out, tuple(task["preview_size"]))
        save_jpeg(out_prev, prev, source_path=None, quality=int(task["jpeg_quality"]))
        save_s = time.perf_counter() - t_save0

        cpu_user = cpu_u1 - cpu_u0
        cpu_sys = cpu_s1 - cpu_s0
        cpu_total = cpu_user + cpu_sys
        peak = max(sampler.peak_mb, rss_after, rss_before)

        result.update(
            {
                "wall_s": round(wall_s, 4),
                "io_s": round(load_s + save_s, 4),
                "total_s": round(wall_s + load_s + save_s, 4),
                "cpu_user_s": round(cpu_user, 4),
                "cpu_sys_s": round(cpu_sys, 4),
                "cpu_total_s": round(cpu_total, 4),
                "cpu_util": round(cpu_total / wall_s, 4) if wall_s > 1e-6 else None,
                "rss_before_mb": round(rss_before, 2),
                "rss_after_mb": round(rss_after, 2),
                "rss_delta_mb": round(rss_after - rss_before, 2),
                "peak_rss_mb": round(peak, 2),
                "peak_rss_delta_mb": round(peak - rss_before, 2),
            }
        )
    except Exception as e:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{e}\n{traceback.format_exc()}"
    return result


def _expand_sweep(cfg: dict, methods: list[str]) -> list[tuple[str, dict, str]]:
    """Return list of (method_name_or_alias, params, out_subdir)."""
    sweep = cfg.get("sweep") or {}
    items: list[tuple[str, dict, str]] = []
    for m in methods:
        base = dict(cfg.get("methods", {}).get(m, {}) or {})
        if m not in sweep:
            items.append((m, base, m))
            continue
        grid = sweep[m]
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        from itertools import product

        for combo in product(*values):
            params = dict(base)
            tag_parts = []
            for k, v in zip(keys, combo):
                params[k] = v
                tag_parts.append(f"{k}={v}")
            alias = m + "__" + "_".join(tag_parts).replace(" ", "").replace(",", "-")
            items.append((m, params, alias))
    return items


def build_tasks(
    manifest: pd.DataFrame,
    cfg: dict,
    run_dir: Path,
    methods: list[str] | None,
    datasets: list[str] | None,
    sweep: bool,
) -> list[dict]:
    shared = cfg.get("shared", {})
    preview_size = tuple(shared.get("preview_size", [2048, 1024]))
    jpeg_q = int(shared.get("jpeg_quality", 95))
    skip = bool(cfg.get("runner", {}).get("skip_existing", True))

    if methods is None or methods == ["all"]:
        methods = list_methods()
    if datasets is not None and datasets != ["all"]:
        manifest = manifest[manifest["dataset"].isin(datasets)]

    if sweep:
        method_specs = _expand_sweep(cfg, methods)
    else:
        method_specs = [(m, dict(cfg.get("methods", {}).get(m, {}) or {}), m) for m in methods]

    tasks = []
    for row in manifest.itertuples(index=False):
        for method_name, params, out_name in method_specs:
            tasks.append(
                {
                    "dataset": row.dataset,
                    "frame_id": row.frame_id,
                    "path": row.path,
                    "method": method_name,
                    "params": params,
                    "shared": shared,
                    "out_full": str(run_dir / "images" / out_name / row.dataset / f"{row.frame_id}.jpg"),
                    "out_preview": str(run_dir / "previews" / out_name / row.dataset / f"{row.frame_id}.jpg"),
                    "preview_size": list(preview_size),
                    "jpeg_quality": jpeg_q,
                    "skip_existing": skip,
                }
            )
    return tasks


def determinism_check(manifest: pd.DataFrame, cfg: dict, n: int = 3) -> list[str]:
    """Run each method twice on n random frames; return list of failure messages."""
    sample = manifest.sample(n=min(n, len(manifest)), random_state=42)
    shared = cfg.get("shared", {})
    failures = []
    for row in sample.itertuples(index=False):
        img = load_image(Path(row.path))
        import cv2

        small = cv2.resize(img, (512, 256), interpolation=cv2.INTER_AREA)
        for m in list_methods():
            if m == "original":
                continue
            params = dict(cfg.get("methods", {}).get(m, {}) or {})
            core = get_method(m)
            a = apply_shared_pipeline(small, core, params, shared)
            b = apply_shared_pipeline(small, core, params, shared)
            if not np.array_equal(a, b):
                failures.append(f"{m} on {row.dataset}/{row.frame_id}: arrays differ")
    return failures


def _append_runtime_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNTIME_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RUNTIME_FIELDS})


def _write_method_summary(runtime_csv: Path, out_csv: Path, out_md: Path) -> None:
    if not runtime_csv.exists():
        return
    df = pd.read_csv(runtime_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    num_cols = [
        "wall_s",
        "io_s",
        "total_s",
        "cpu_user_s",
        "cpu_sys_s",
        "cpu_total_s",
        "cpu_util",
        "peak_rss_mb",
        "peak_rss_delta_mb",
        "rss_delta_mb",
        "megapixels",
    ]
    for c in num_cols:
        if c in ok.columns:
            ok[c] = pd.to_numeric(ok[c], errors="coerce")

    rows = []
    for method, g in ok.groupby("method"):
        row = {
            "method": method,
            "n_ok": len(g),
            "n_failed": int(((df["method"] == method) & (df["status"] == "failed")).sum()),
            "n_skipped": int(((df["method"] == method) & (df["status"] == "skipped")).sum()),
            "wall_s_median": g["wall_s"].median(),
            "wall_s_mean": g["wall_s"].mean(),
            "wall_s_p95": g["wall_s"].quantile(0.95),
            "wall_s_sum": g["wall_s"].sum(),
            "cpu_total_s_median": g["cpu_total_s"].median(),
            "cpu_total_s_sum": g["cpu_total_s"].sum(),
            "cpu_util_median": g["cpu_util"].median(),
            "peak_rss_mb_median": g["peak_rss_mb"].median(),
            "peak_rss_mb_max": g["peak_rss_mb"].max(),
            "peak_rss_delta_mb_median": g["peak_rss_delta_mb"].median(),
            "s_per_mp_median": (g["wall_s"] / g["megapixels"].clip(lower=1e-6)).median(),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("wall_s_median")
    summary.to_csv(out_csv, index=False)

    # Prefer plain text — pandas.to_markdown() requires optional `tabulate`
    try:
        table = summary.to_markdown(index=False)
    except Exception:  # noqa: BLE001
        table = summary.to_string(index=False)

    lines = [
        "# Per-method resource summary",
        "",
        f"Updated: {datetime.now(timezone.utc).isoformat()}",
        f"Rows in runtime.csv: {len(df)} (ok={len(ok)})",
        "",
        "```",
        table,
        "```",
        "",
        "Notes:",
        "- `wall_s`: enhancement only (excludes JPEG load/save).",
        "- `cpu_util`: cpu_total_s / wall_s (≈1.0 means one core saturated; >1 means multi-threaded).",
        "- `peak_rss_mb`: sampled process RSS peak during enhancement.",
        "- `peak_rss_delta_mb`: peak − RSS just before enhancement (allocation attributed to the method).",
        "",
    ]
    out_md.write_text("\n".join(lines))


class _FlushFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def _setup_run_logging(run_dir: Path) -> Path:
    log_path = run_dir / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    fh = _FlushFileHandler(log_path)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    # Console mirror (may be redirected via nohup); never let it kill the run
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run enhancement methods")
    parser.add_argument("--zips", type=Path, default=ROOT / "data" / "zips")
    parser.add_argument("--frames", type=Path, default=ROOT / "data" / "frames")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks (debug)")
    parser.add_argument(
        "--summary-every",
        type=int,
        default=25,
        help="Rewrite per-method summary every N completed ok/failed tasks",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if not args.manifest.exists():
        logging.basicConfig(level=logging.INFO)
        log.info("building manifest…")
        ingest(args.zips, args.frames, args.manifest)
    manifest = pd.read_csv(args.manifest)

    run_id = args.run_id or make_run_id(cfg)
    run_dir = ROOT / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _setup_run_logging(run_dir)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Hardware snapshot for the report
    hw = {
        "cpu_model": _cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "workers": args.workers or worker_count(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "hardware.json").write_text(json.dumps(hw, indent=2))
    log.info("hardware: %s", hw)
    log.info("log file: %s", log_path)

    if not args.skip_determinism:
        log.info("determinism check…")
        fails = determinism_check(manifest, cfg)
        if fails:
            log.error("determinism failures:\n%s", "\n".join(fails))
            (run_dir / "determinism_failures.txt").write_text("\n".join(fails))
        else:
            log.info("determinism OK")
            (run_dir / "determinism_ok.txt").write_text("ok\n")

    tasks = build_tasks(
        manifest,
        cfg,
        run_dir,
        None if args.methods == ["all"] else args.methods,
        None if args.datasets == ["all"] else args.datasets,
        sweep=args.sweep,
    )
    if args.limit:
        tasks = tasks[: args.limit]

    n_workers = args.workers or worker_count()
    runtime_csv = run_dir / "runtime.csv"
    summary_csv = run_dir / "logs" / "per_method_summary.csv"
    summary_md = run_dir / "logs" / "per_method_summary.md"
    events_log = run_dir / "logs" / "events.jsonl"

    log.info("run_id=%s  tasks=%d  workers=%d", run_id, len(tasks), n_workers)
    log.info("runtime csv: %s", runtime_csv)

    n_ok = n_fail = n_skip = 0
    done_since_summary = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(process_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="enhance"):
            row = fut.result()
            _append_runtime_row(runtime_csv, row)
            with open(events_log, "a") as ef:
                ef.write(json.dumps(row, default=str) + "\n")

            status = row.get("status")
            if status == "ok":
                n_ok += 1
                done_since_summary += 1
                log.info(
                    "OK %s/%s [%s] wall=%.2fs cpu=%.2fs util=%.2f peak_rss=%.0fMB (+%.0f)",
                    row["dataset"],
                    row["frame_id"],
                    row["method"],
                    row.get("wall_s") or 0,
                    row.get("cpu_total_s") or 0,
                    row.get("cpu_util") or 0,
                    row.get("peak_rss_mb") or 0,
                    row.get("peak_rss_delta_mb") or 0,
                )
            elif status == "skipped":
                n_skip += 1
                log.debug("SKIP %s/%s [%s]", row["dataset"], row["frame_id"], row["method"])
            else:
                n_fail += 1
                done_since_summary += 1
                err = (row.get("error") or "").split("\n")[0]
                log.error("FAIL %s/%s [%s]: %s", row["dataset"], row["frame_id"], row["method"], err)

            if done_since_summary >= args.summary_every:
                try:
                    _write_method_summary(runtime_csv, summary_csv, summary_md)
                except Exception as e:  # noqa: BLE001
                    log.exception("summary refresh failed (continuing): %s", e)
                done_since_summary = 0

    try:
        _write_method_summary(runtime_csv, summary_csv, summary_md)
    except Exception as e:  # noqa: BLE001
        log.exception("final summary failed: %s", e)
    hw["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "hardware.json").write_text(json.dumps(hw, indent=2))

    log.info(
        "done. ok=%d failed=%d skipped=%d run_dir=%s",
        n_ok,
        n_fail,
        n_skip,
        run_dir,
    )
    if summary_md.exists():
        log.info("per-method summary:\n%s", summary_md.read_text())
    print(run_dir)
    return 0 if n_fail == 0 else 1


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
