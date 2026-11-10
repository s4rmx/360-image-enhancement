"""Streamlit multipage UI for the 360 enhance research benchmark."""

from __future__ import annotations

import argparse
import base64
import io
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enhance.equirect import seam_crop  # noqa: E402
from enhance.io import load_image  # noqa: E402

METHODS = ["original", "wb_only", "clahe", "msrcr", "lime", "fusion", "llf"]
ENHANCE = ["wb_only", "clahe", "msrcr", "lime", "fusion", "llf"]


def _parse_run() -> Path:
    # streamlit run app/app.py -- --run outputs/<id>
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", type=str, default="")
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.run:
        return Path(args.run)
    # latest run
    out = ROOT / "outputs"
    if out.exists():
        runs = sorted([p for p in out.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            return runs[0]
    return out / "NONE"


def _ensure_votes_db(run_dir: Path) -> Path:
    db = run_dir / "votes.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voter TEXT,
            ts TEXT,
            dataset TEXT,
            frame_id TEXT,
            method_a TEXT,
            method_b TEXT,
            choice TEXT
        )
        """
    )
    con.commit()
    con.close()
    return db


def _preview_path(run_dir: Path, method: str, dataset: str, frame_id: str) -> Path:
    return run_dir / "previews" / method / dataset / f"{frame_id}.jpg"


def _full_path(run_dir: Path, method: str, dataset: str, frame_id: str) -> Path:
    return run_dir / "images" / method / dataset / f"{frame_id}.jpg"


def _list_frames(run_dir: Path) -> pd.DataFrame:
    rows = []
    prev = run_dir / "previews" / "original"
    if not prev.exists():
        # fall back to any method
        methods = [p for p in (run_dir / "previews").iterdir() if p.is_dir()] if (run_dir / "previews").exists() else []
        prev = methods[0] if methods else None
    if prev is None:
        return pd.DataFrame(columns=["dataset", "frame_id"])
    for ds in sorted(p for p in prev.iterdir() if p.is_dir()):
        for f in sorted(ds.glob("*.jpg")):
            rows.append({"dataset": ds.name, "frame_id": f.stem})
    return pd.DataFrame(rows)


def _metrics_for(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "metrics.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


def page_browse(run_dir: Path) -> None:
    st.header("Browse / Compare")
    frames = _list_frames(run_dir)
    if frames.empty:
        st.warning("No previews found in this run.")
        return
    datasets = sorted(frames["dataset"].unique())
    dataset = st.selectbox("Dataset", datasets)
    frame_ids = frames.loc[frames["dataset"] == dataset, "frame_id"].tolist()
    frame_id = st.selectbox("Frame", frame_ids)
    metrics = _metrics_for(run_dir)

    cols = st.columns(4)
    for i, m in enumerate(METHODS):
        path = _preview_path(run_dir, m, dataset, frame_id)
        with cols[i % 4]:
            st.caption(m)
            if path.exists():
                st.image(str(path), use_container_width=True)
                if not metrics.empty:
                    row = metrics[
                        (metrics.dataset == dataset) & (metrics.frame_id == frame_id) & (metrics.method == m)
                    ]
                    if len(row):
                        r = row.iloc[0]
                        st.caption(
                            f"NIQE {r.get('niqe', float('nan')):.2f} · "
                            f"clip {r.get('clipping_pct', float('nan')):.1f}% · "
                            f"seam {r.get('seam_delta', float('nan')):.2f}"
                        )
            else:
                st.write("_missing_")

    st.subheader("Before / After")
    method = st.selectbox("Compare method", [m for m in METHODS if m != "original"])
    try:
        from streamlit_image_comparison import image_comparison

        a = _preview_path(run_dir, "original", dataset, frame_id)
        b = _preview_path(run_dir, method, dataset, frame_id)
        if a.exists() and b.exists():
            image_comparison(img1=str(a), img2=str(b), label1="original", label2=method)
    except Exception as e:  # noqa: BLE001
        st.info(f"image-comparison unavailable ({e}); showing side-by-side")
        c1, c2 = st.columns(2)
        c1.image(str(_preview_path(run_dir, "original", dataset, frame_id)), caption="original")
        c2.image(str(_preview_path(run_dir, method, dataset, frame_id)), caption=method)

    st.subheader("Seam strips")
    for m in METHODS:
        p = _preview_path(run_dir, m, dataset, frame_id)
        if not p.exists():
            continue
        img = load_image(p)
        strip = seam_crop(img, k=128)
        st.caption(m)
        st.image(np.clip(strip, 0, 1), use_container_width=True, clamp=True)

    st.subheader("Full-res crop")
    cx = st.number_input("x", min_value=0, value=0)
    cy = st.number_input("y", min_value=0, value=0)
    sz = st.number_input("size", min_value=64, value=512)
    crop_cols = st.columns(len(METHODS))
    for i, m in enumerate(METHODS):
        p = _full_path(run_dir, m, dataset, frame_id)
        with crop_cols[i]:
            st.caption(m)
            if p.exists():
                img = load_image(p)
                h, w = img.shape[:2]
                x0, y0 = int(cx) % w, int(cy) % h
                patch = img[y0 : y0 + int(sz), x0 : x0 + int(sz)]
                if patch.size:
                    st.image(np.clip(patch, 0, 1), use_container_width=True, clamp=True)


def _pannellum_html(img_b64: str, yaw: float = 0.0, pitch: float = 0.0) -> str:
    return f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.css"/>
    <script src="https://cdn.jsdelivr.net/npm/pannellum@2.5.6/build/pannellum.js"></script>
    <div id="panorama" style="width:100%;height:480px;"></div>
    <script>
      pannellum.viewer('panorama', {{
        type: 'equirectangular',
        panorama: 'data:image/jpeg;base64,{img_b64}',
        autoLoad: true,
        yaw: {yaw},
        pitch: {pitch},
        hfov: 100
      }});
    </script>
    """


def page_360(run_dir: Path) -> None:
    st.header("360 Viewer")
    frames = _list_frames(run_dir)
    if frames.empty:
        st.warning("No frames.")
        return
    dataset = st.selectbox("Dataset", sorted(frames.dataset.unique()), key="v_ds")
    frame_id = st.selectbox(
        "Frame", frames.loc[frames.dataset == dataset, "frame_id"].tolist(), key="v_fr"
    )
    method = st.selectbox("Method", METHODS, key="v_m")
    yaw = st.session_state.get("yaw", 0.0)
    pitch = st.session_state.get("pitch", 0.0)
    path = _preview_path(run_dir, method, dataset, frame_id)
    if not path.exists():
        st.error("Missing preview")
        return
    # Cap at 4096x2048
    im = Image.open(path)
    if im.width > 4096:
        im = im.resize((4096, 2048), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    import streamlit.components.v1 as components

    components.html(_pannellum_html(b64, yaw=yaw, pitch=pitch), height=500)


def page_vote(run_dir: Path) -> None:
    st.header("Blind Vote")
    db = _ensure_votes_db(run_dir)
    voter = st.text_input("Voter name", value=st.session_state.get("voter", ""))
    st.session_state["voter"] = voter
    use_360 = st.toggle("Use 360 viewer", value=False)

    frames = _list_frames(run_dir)
    if frames.empty:
        st.warning("No frames.")
        return

    if "vote_pair" not in st.session_state:
        _resample_pair(frames)

    pair = st.session_state["vote_pair"]
    st.caption(f"{pair['dataset']} / {pair['frame_id']}")

    orig = _preview_path(run_dir, "original", pair["dataset"], pair["frame_id"])
    pa = _preview_path(run_dir, pair["method_a"], pair["dataset"], pair["frame_id"])
    pb = _preview_path(run_dir, pair["method_b"], pair["dataset"], pair["frame_id"])

    st.subheader("Original")
    if orig.exists():
        st.image(str(orig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A")
        if pa.exists():
            if use_360:
                _show_mini_pano(pa)
            else:
                st.image(str(pa), use_container_width=True)
    with c2:
        st.subheader("B")
        if pb.exists():
            if use_360:
                _show_mini_pano(pb)
            else:
                st.image(str(pb), use_container_width=True)

    b1, b2, b3, b4 = st.columns(4)
    choice = None
    if b1.button("A better"):
        choice = "A"
    if b2.button("B better"):
        choice = "B"
    if b3.button("Tie"):
        choice = "Tie"
    if b4.button("Both worse than original"):
        choice = "BothWorse"

    if choice:
        # Map displayed A/B back if we shuffled labels — stored methods already match display
        con = sqlite3.connect(db)
        con.execute(
            "INSERT INTO votes (voter, ts, dataset, frame_id, method_a, method_b, choice) VALUES (?,?,?,?,?,?,?)",
            (
                voter,
                datetime.now(timezone.utc).isoformat(),
                pair["dataset"],
                pair["frame_id"],
                pair["method_a"],
                pair["method_b"],
                choice,
            ),
        )
        con.commit()
        con.close()
        st.success(f"Recorded: {choice}")
        _resample_pair(frames)
        st.rerun()

    n = pd.read_sql_query("SELECT COUNT(*) AS n FROM votes", sqlite3.connect(db)).iloc[0]["n"]
    st.caption(f"Total votes in this run: {n}")


def _show_mini_pano(path: Path) -> None:
    im = Image.open(path)
    if im.width > 2048:
        im = im.resize((2048, 1024), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    import streamlit.components.v1 as components

    components.html(_pannellum_html(b64), height=320)


def _resample_pair(frames: pd.DataFrame) -> None:
    rng = np.random.default_rng()
    row = frames.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
    methods = list(ENHANCE)
    a, b = rng.choice(methods, size=2, replace=False)
    # Randomize display order already via choice
    if rng.random() < 0.5:
        a, b = b, a
    st.session_state["vote_pair"] = {
        "dataset": row.dataset,
        "frame_id": row.frame_id,
        "method_a": str(a),
        "method_b": str(b),
    }


def page_results(run_dir: Path) -> None:
    st.header("Results")
    metrics = _metrics_for(run_dir)
    if metrics.empty:
        st.warning("No metrics.csv — run `python -m enhance.benchmark --run ...`")
        return

    scope = st.radio("Scope", ["overall", "per dataset"], horizontal=True)
    import plotly.express as px

    if scope == "overall":
        agg = metrics.groupby("method").median(numeric_only=True).reset_index()
        st.dataframe(agg, use_container_width=True)
    else:
        ds = st.selectbox("Dataset", sorted(metrics.dataset.unique()))
        agg = metrics[metrics.dataset == ds].groupby("method").median(numeric_only=True).reset_index()
        st.dataframe(agg, use_container_width=True)

    metric = st.selectbox(
        "Box plot metric",
        [c for c in metrics.columns if c not in {"dataset", "frame_id", "method"} and pd.api.types.is_numeric_dtype(metrics[c])],
    )
    fig = px.box(metrics, x="method", y=metric, points="outliers")
    st.plotly_chart(fig, use_container_width=True)

    db = run_dir / "votes.sqlite"
    if db.exists():
        votes = pd.read_sql_query("SELECT * FROM votes", sqlite3.connect(db))
        st.subheader("Votes")
        st.dataframe(votes, use_container_width=True)
        if len(votes):
            from enhance.report import vote_stats

            win, pair, bw = vote_stats(votes)
            st.write("Win rate")
            st.dataframe(win)
            st.write("Pairwise")
            st.dataframe(pair)
            st.write("Both-worse rate")
            st.dataframe(bw.to_frame("both_worse_rate"))

    rt = run_dir / "runtime.csv"
    if rt.exists():
        rdf = pd.read_csv(rt)
        st.subheader("Runtime / RAM")
        st.dataframe(rdf.groupby("method")[["seconds", "peak_rss_mb"]].median(), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="360 Enhance Research", layout="wide")
    run_dir = _parse_run()
    st.sidebar.title("360 Enhance")
    st.sidebar.write(f"Run: `{run_dir}`")
    if not run_dir.exists():
        st.error("Run directory not found. Pass `-- --run outputs/<run_id>`.")
        return
    page = st.sidebar.radio("Page", ["Browse / Compare", "360 Viewer", "Blind Vote", "Results"])
    if page == "Browse / Compare":
        page_browse(run_dir)
    elif page == "360 Viewer":
        page_360(run_dir)
    elif page == "Blind Vote":
        page_vote(run_dir)
    else:
        page_results(run_dir)


if __name__ == "__main__":
    main()
