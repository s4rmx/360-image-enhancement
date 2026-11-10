#!/usr/bin/env python3
"""Generate an HTML original-vs-CLAHE comparison report for selected datasets."""

from __future__ import annotations

import argparse
import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def collect_frames(run_dir: Path, dataset: str) -> list[str]:
    clahe_dir = run_dir / "previews" / "clahe" / dataset
    orig_dir = run_dir / "previews" / "original" / dataset
    if not clahe_dir.exists() or not orig_dir.exists():
        return []
    frames = sorted(p.stem for p in clahe_dir.glob("*.jpg") if (orig_dir / p.name).exists())
    return frames


def runtime_stats(run_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rt = run_dir / "runtime.csv"
    if not rt.exists():
        return pd.DataFrame()
    df = pd.read_csv(rt)
    df = df[(df["status"] == "ok") & (df["method"] == "clahe") & (df["dataset"].isin(datasets))]
    if df.empty:
        return df
    return (
        df.groupby("dataset")
        .agg(
            n=("frame_id", "count"),
            wall_s_median=("wall_s", "median"),
            wall_s_mean=("wall_s", "mean"),
            peak_rss_mb_median=("peak_rss_mb", "median"),
            peak_rss_mb_max=("peak_rss_mb", "max"),
            megapixels=("megapixels", "median"),
        )
        .reset_index()
    )


def build_html(run_dir: Path, datasets: list[str], out_path: Path, max_frames: int | None) -> None:
    run_dir = run_dir.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Paths in HTML relative to the report file
    rel_run = Path(os_path_rel(out_path.parent, run_dir))

    stats = runtime_stats(run_dir, datasets)
    sections = []
    total_frames = 0

    for dataset in datasets:
        frames = collect_frames(run_dir, dataset)
        if max_frames is not None:
            frames = frames[:max_frames]
        total_frames += len(frames)
        cards = []
        for fid in frames:
            orig = rel_run / "previews" / "original" / dataset / f"{fid}.jpg"
            clahe = rel_run / "previews" / "clahe" / dataset / f"{fid}.jpg"
            cards.append(
                f"""
        <article class="card" id="{html.escape(dataset)}__{html.escape(fid)}">
          <header><h3>{html.escape(fid)}</h3></header>
          <div class="pair">
            <figure>
              <img loading="lazy" src="{html.escape(orig.as_posix())}" alt="original {html.escape(fid)}"/>
              <figcaption>Original</figcaption>
            </figure>
            <figure>
              <img loading="lazy" src="{html.escape(clahe.as_posix())}" alt="clahe {html.escape(fid)}"/>
              <figcaption>CLAHE</figcaption>
            </figure>
          </div>
        </article>"""
            )

        ds_stat = ""
        if not stats.empty:
            row = stats[stats["dataset"] == dataset]
            if len(row):
                r = row.iloc[0]
                ds_stat = (
                    f"<p class='meta'>{int(r.n)} frames · "
                    f"median {r.wall_s_median:.2f}s · "
                    f"peak RSS med {r.peak_rss_mb_median:.0f} MB · "
                    f"{r.megapixels:.1f} MP</p>"
                )

        toc_link = f'<a href="#ds-{html.escape(dataset)}">{html.escape(dataset)}</a> ({len(frames)})'
        sections.append(
            (
                toc_link,
                f"""
      <section class="dataset" id="ds-{html.escape(dataset)}">
        <h2>{html.escape(dataset)}</h2>
        {ds_stat}
        <div class="grid">
          {''.join(cards)}
        </div>
      </section>""",
            )
        )

    toc = " · ".join(t for t, _ in sections)
    body = "\n".join(b for _, b in sections)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CLAHE vs Original — locked method report</title>
  <style>
    :root {{
      --bg: #0f1216;
      --panel: #171b22;
      --text: #e8eaed;
      --muted: #9aa3ad;
      --accent: #6cb6ff;
      --line: #2a313c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 600px at 10% -10%, #1a2433, var(--bg));
      color: var(--text);
      line-height: 1.45;
    }}
    header.hero {{
      padding: 2.5rem 1.5rem 1.5rem;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(108,182,255,0.08), transparent);
    }}
    header.hero h1 {{ margin: 0 0 0.4rem; font-size: 1.8rem; letter-spacing: -0.02em; }}
    header.hero p {{ margin: 0.25rem 0; color: var(--muted); max-width: 70ch; }}
    .badge {{
      display: inline-block;
      margin-top: 0.8rem;
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      background: #1e3a5f;
      color: var(--accent);
      font-size: 0.85rem;
      font-weight: 600;
    }}
    nav.toc {{
      padding: 1rem 1.5rem;
      position: sticky; top: 0; z-index: 5;
      background: rgba(15,18,22,0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--line);
      font-size: 0.95rem;
    }}
    nav.toc a {{ color: var(--accent); text-decoration: none; margin-right: 0.25rem; }}
    nav.toc a:hover {{ text-decoration: underline; }}
    main {{ padding: 1.5rem; max-width: 1400px; margin: 0 auto; }}
    section.dataset {{ margin-bottom: 3rem; }}
    section.dataset h2 {{
      font-size: 1.35rem;
      margin: 0 0 0.4rem;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid var(--line);
    }}
    .meta {{ color: var(--muted); margin: 0 0 1rem; font-size: 0.92rem; }}
    .grid {{ display: flex; flex-direction: column; gap: 1.25rem; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
    }}
    .card header {{
      padding: 0.65rem 1rem;
      border-bottom: 1px solid var(--line);
    }}
    .card h3 {{ margin: 0; font-size: 0.95rem; font-weight: 600; color: var(--muted); }}
    .pair {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }}
    figure {{ margin: 0; background: #000; }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 2 / 1;
      object-fit: cover;
    }}
    figcaption {{
      padding: 0.45rem 0.75rem;
      font-size: 0.8rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      border-top: 1px solid var(--line);
      background: #12161c;
    }}
    footer {{
      padding: 2rem 1.5rem 3rem;
      color: var(--muted);
      font-size: 0.85rem;
      border-top: 1px solid var(--line);
      text-align: center;
    }}
    @media (max-width: 800px) {{
      .pair {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>CLAHE vs Original</h1>
    <p>Locked enhancement method for 360° equirectangular frames. Side-by-side previews
       (2048×1024) from research run <code>{html.escape(run_dir.name)}</code>.</p>
    <span class="badge">Method locked: CLAHE</span>
    <p style="margin-top:0.8rem">Datasets: {html.escape(', '.join(datasets))} · {total_frames} frames · generated {generated}</p>
  </header>
  <nav class="toc">{toc}</nav>
  <main>
{body}
  </main>
  <footer>
    360 Image Auto-Enhance research · deterministic CLAHE (clipLimit=2.0, grid 8×16) with shared WB + post-levels
  </footer>
</body>
</html>
"""
    out_path.write_text(doc)
    print(out_path)


def os_path_rel(from_dir: Path, to_path: Path) -> str:
    import os

    return os.path.relpath(to_path, from_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["6th-Oct-First-Floor", "Basement - 1-1545dc10", "Walk 1"],
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    out = args.out or (args.run / "reports" / "clahe_vs_original.html")
    build_html(args.run, args.datasets, out, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
