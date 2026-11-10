"""CLI: compute metrics over a completed run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from enhance.io import load_image
from enhance.metrics import compute_metrics

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute metrics for a run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument("--skip-iqa", action="store_true", help="Skip NIQE/BRISQUE (faster)")
    parser.add_argument("--methods", nargs="+", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = Path(args.run)
    preview_root = run_dir / "previews"
    image_root = run_dir / "images"
    manifest = pd.read_csv(args.manifest)

    methods = args.methods or sorted(p.name for p in preview_root.iterdir() if p.is_dir())
    rows = []
    for method in methods:
        for row in tqdm(list(manifest.itertuples(index=False)), desc=method):
            prev_path = preview_root / method / row.dataset / f"{row.frame_id}.jpg"
            full_path = image_root / method / row.dataset / f"{row.frame_id}.jpg"
            orig_prev = preview_root / "original" / row.dataset / f"{row.frame_id}.jpg"
            if not prev_path.exists():
                continue
            # Original preview may be missing if original wasn't run; build from source
            enhanced = load_image(prev_path)
            if orig_prev.exists():
                original = load_image(orig_prev)
            else:
                original = load_image(Path(row.path))
                from enhance.io import resize_preview

                original = resize_preview(original)

            full_e = load_image(full_path) if full_path.exists() else None
            full_o = load_image(Path(row.path)) if full_path.exists() else None

            m = compute_metrics(
                enhanced,
                original,
                full_enhanced=full_e,
                full_original=full_o,
                skip_iqa=args.skip_iqa,
            )
            m.update({"dataset": row.dataset, "frame_id": row.frame_id, "method": method})
            rows.append(m)

    df = pd.DataFrame(rows)
    # Attach runtime if available
    rt_path = run_dir / "runtime.csv"
    if rt_path.exists() and not df.empty:
        rt = pd.read_csv(rt_path)
        rt = rt[rt["status"] == "ok"][["dataset", "frame_id", "method", "seconds", "peak_rss_mb"]]
        df = df.merge(rt, on=["dataset", "frame_id", "method"], how="left")

    out_parquet = run_dir / "metrics.parquet"
    out_csv = run_dir / "metrics.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    log.info("wrote %s (%d rows)", out_csv, len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
