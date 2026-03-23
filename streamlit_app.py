from datetime import datetime
from html import escape
import io
import json
import os
import ssl
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Dict, cast
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import streamlit as st  # type: ignore[import-not-found]
import streamlit.components.v1 as components  # type: ignore[import-not-found]

from predict_disease_probabilities import run_inference


def is_normal_disease_name(name: str) -> bool:
    text = str(name).strip().lower()
    return text == "normal/other unspecified disease" or text.startswith("normal")


def normalize_sample_id(value: Any) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        return text[2:-1]
    return text


def parse_abnormal_markers(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if ":" in text:
        text = text.split(":", 1)[1]

    return [marker.strip() for marker in text.split(";") if marker.strip()]


def extract_top1_marker_and_value(row: pd.Series) -> tuple[str, str]:
    markers = parse_abnormal_markers(row.get("Abnormals", ""))
    if not markers:
        return "-", "-"

    marker_name = markers[0]
    marker_value = row.get(marker_name, "-")
    if pd.isna(marker_value) or str(marker_value).strip() == "":
        marker_value_text = "-"
    else:
        marker_value_text = str(marker_value)

    return marker_name, marker_value_text


def build_marker_info_table(input_df: pd.DataFrame, source_id_col: str) -> pd.DataFrame:
    marker_df = pd.DataFrame({"sample_id": input_df[source_id_col].map(normalize_sample_id)})

    if "Abnormals" not in input_df.columns:
        marker_df["top_1_marker"] = "-"
        marker_df["top_1_marker_value"] = "-"
        return marker_df

    marker_pairs = input_df.apply(extract_top1_marker_and_value, axis=1)
    marker_df["top_1_marker"] = [pair[0] for pair in marker_pairs]
    marker_df["top_1_marker_value"] = [pair[1] for pair in marker_pairs]
    return marker_df


def load_local_env(env_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from .env into process env if not already set."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def get_discord_webhook() -> str:
    # Prefer Streamlit secrets (cloud), then fallback to local env/.env.
    try:
        webhook_from_secrets = str(st.secrets.get("DISCORD_WEBHOOK_URL", "")).strip()
        if webhook_from_secrets:
            return webhook_from_secrets
    except Exception:
        pass

    return os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def send_discord_message(webhook_url: str, content: str) -> None:
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urlrequest.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ssl_context = None
    try:
        import certifi  # type: ignore[import-not-found]

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ssl_context = ssl.create_default_context()

    try:
        with urlrequest.urlopen(req, timeout=15, context=ssl_context) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"Discord webhook failed with status {resp.status}")
    except ssl.SSLCertVerificationError as exc:
        # Fallback to curl because macOS Python SSL trust stores can be inconsistent.
        _send_discord_via_curl(webhook_url, content, reason=str(exc))
    except Exception as exc:
        # Generic transport fallback path.
        _send_discord_via_curl(webhook_url, content, reason=str(exc))


def _send_discord_via_curl(webhook_url: str, content: str, reason: str = "") -> None:
    payload = json.dumps({"content": content})
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        "20",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-H",
        "Content-Type: application/json",
        "-d",
        payload,
        webhook_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    code = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "Discord send failed via urllib and curl. "
            f"curl rc={proc.returncode}. Details: {proc.stderr.strip() or reason}"
        )
    if code not in ("200", "204"):
        raise RuntimeError(
            "Discord send failed via urllib and curl. "
            f"curl HTTP={code or 'unknown'}. Details: {reason}"
        )


def send_discord_file_attachment(
    webhook_url: str,
    file_path: str,
    content: str = "",
    mime_type: str = "application/octet-stream",
) -> None:
    payload_json = json.dumps({"content": content})
    cmd = [
        "curl",
        "-sS",
        "--max-time",
        "30",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "-F",
        f"payload_json={payload_json}",
        "-F",
        f"file=@{file_path};type={mime_type}",
        webhook_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    code = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "Discord attachment send failed. "
            f"curl rc={proc.returncode}. Details: {proc.stderr.strip()}"
        )
    if code not in ("200", "204"):
        raise RuntimeError(f"Discord attachment send failed. curl HTTP={code or 'unknown'}")


def build_discord_suspected_summary(df: pd.DataFrame, id_col: str) -> str:
    suspected = get_patient_suspected_cases(df, id_col)
    if suspected.empty:
        return "IEM screening report: no suspected disease cases (top-1) found."

    ids = ", ".join(suspected[id_col].astype(str).tolist())
    lines = [
        "IEM screening report (suspected disease cases only)",
        f"Count: {len(suspected)}",
        f"Sample IDs: {ids}",
        "",
        "Top-1 predictions:",
    ]

    for _, row in suspected.head(20).iterrows():
        top_marker = str(row.get("top_1_marker", "-") or "-")
        top_marker_value = str(row.get("top_1_marker_value", "-") or "-")
        lines.append(
            f"- {row[id_col]} | {row['top_1_disease']} | {row['top_1_probability']*100:.2f}%"
            f" | Marker: {top_marker} = {top_marker_value}"
        )

    if len(suspected) > 20:
        lines.append(f"... and {len(suspected)-20} more rows")

    message = "\n".join(lines)
    return message[:1900]


def build_suspected_html_report(df: pd.DataFrame, id_col: str, output_path: str) -> str:
    suspected = get_patient_suspected_cases(df, id_col)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if suspected.empty:
        html = f"""
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"utf-8\" />
            <title>Suspected Disease Cases Report</title>
            <style>
                body {{ font-family: Arial, Helvetica, sans-serif; margin: 24px 32px; color: #222; }}
                .page-title {{ background: #1e8f4e; color: #fff; text-align: center; font-size: 22px; font-weight: 700; padding: 12px; }}
                .ok {{ margin-top: 16px; padding: 10px 12px; border: 1px solid #a6dcb7; background: #eefaf1; color: #18633a; border-radius: 8px; }}
                .footer {{ margin-top: 20px; color: #8a8a8a; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class=\"page-title\">Disease Pattern Detection Report (Suspected Cases Only)</div>
            <div class=\"ok\"><strong>No suspected disease cases found.</strong></div>
            <div class=\"footer\">Generated: {generated_at}</div>
        </body>
        </html>
        """
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    suspect_ids = ", ".join(suspected[id_col].astype(str).tolist())
    rows = []
    for _, row in suspected.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row[id_col]))}</td>"
            f"<td>{escape(str(row.get('top_1_disease', '-')))}</td>"
            f"<td>{float(row.get('top_1_probability', 0.0)) * 100:.2f}%</td>"
            f"<td>{escape(str(row.get('top_1_marker', '-')))}</td>"
            f"<td>{escape(str(row.get('top_1_marker_value', '-')))}</td>"
            f"<td>{escape(str(row.get('top_2_disease', '-')))}</td>"
            f"<td>{float(row.get('top_2_probability', 0.0)) * 100:.2f}%</td>"
            f"<td>{escape(str(row.get('top_3_disease', '-')))}</td>"
            f"<td>{float(row.get('top_3_probability', 0.0)) * 100:.2f}%</td>"
            "</tr>"
        )

    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <title>Suspected Disease Cases Report</title>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 24px 32px;
                color: #222;
                background: #fff;
            }}
            .page-title {{
                background: #1e8f4e;
                color: #fff;
                text-align: center;
                font-size: 22px;
                font-weight: 700;
                padding: 12px;
                margin-bottom: 16px;
            }}
            .suspect-summary {{
                margin-bottom: 14px;
                padding: 10px 12px;
                border-radius: 8px;
                border: 1px solid #e59db0;
                background: #fff2f6;
                color: #8f1638;
                font-size: 13px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            thead th {{
                background: #1e8f4e;
                color: #fff;
                border: 2px solid #16673a;
                padding: 8px 10px;
                font-size: 14px;
            }}
            tbody td {{
                border: 1px solid #cde8d5;
                padding: 8px 10px;
                font-size: 14px;
                color: #b00020;
                vertical-align: top;
            }}
            tbody tr:nth-child(1) td {{
                font-weight: 700;
                background: #fdf1f4;
            }}
            .footer {{
                margin-top: 18px;
                color: #8a8a8a;
                font-size: 12px;
            }}
        </style>
    </head>
    <body>
        <div class=\"page-title\">Disease Pattern Detection Report (Suspected Cases Only)</div>
        <div class=\"suspect-summary\"><strong>Suspected disease sample IDs (grouped):</strong> {escape(suspect_ids)}</div>
        <table>
            <thead>
                <tr>
                    <th>Sample ID</th>
                    <th>Top 1 Disease</th>
                    <th>Top 1 Probability</th>
                    <th>Top 1 Marker</th>
                    <th>Marker Value</th>
                    <th>Top 2 Disease</th>
                    <th>Top 2 Probability</th>
                    <th>Top 3 Disease</th>
                    <th>Top 3 Probability</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <div class=\"footer\">Generated: {generated_at}</div>
    </body>
    </html>
    """

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def is_control_or_internal_sample(sample_id: str) -> bool:
    text = str(sample_id).strip().upper()
    if not text:
        return False
    return text.startswith("LC") or text.startswith("HC") or text == "IS"


def get_patient_suspected_cases(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    suspected = df[~df["top_1_disease"].astype(str).apply(is_normal_disease_name)].copy()
    if suspected.empty:
        return suspected
    id_text = suspected[id_col].astype(str)
    patient_mask = ~id_text.apply(is_control_or_internal_sample)
    return suspected.loc[patient_mask].copy()


def build_suspected_png_report(df: pd.DataFrame, id_col: str, output_path: str) -> str:
    suspected = get_patient_suspected_cases(df, id_col)
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for PNG report generation. Install it in the active environment."
        ) from exc

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if suspected.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.75,
            "Disease Pattern Detection Report (Suspected Patient Cases Only)",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#1e8f4e",
        )
        ax.text(
            0.5,
            0.45,
            "No suspected patient cases found.",
            ha="center",
            va="center",
            fontsize=14,
            color="#18633a",
        )
        ax.text(0.5, 0.15, f"Generated: {generated_at}", ha="center", va="center", fontsize=10.5, color="#666")
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path

    view = suspected[[id_col, "top_1_disease", "top_1_probability", "top_1_marker", "top_1_marker_value"]].copy()
    view = view.rename(columns={
        id_col: "Sample ID",
        "top_1_disease": "Top 1 Disease",
        "top_1_probability": "Top 1 %",
        "top_1_marker": "Top 1 Marker",
        "top_1_marker_value": "Marker Value",
    })

    for col in ["Top 1 %"]:
        view[col] = view[col].astype(float).map(lambda v: f"{v * 100:.2f}%")

    max_rows = 20
    clipped = view.head(max_rows)
    fig_h = max(5.8, 1.8 + (len(clipped) * 0.58))
    fig, ax = plt.subplots(figsize=(22, fig_h))
    ax.axis("off")

    title = "Disease Pattern Detection Report (Suspected Patient Cases Only)"
    ids = ", ".join(clipped["Sample ID"].astype(str).tolist())
    ax.text(0.5, 1.09, title, ha="center", va="center", fontsize=19, fontweight="bold", color="#1e8f4e", transform=ax.transAxes)
    ax.text(0.0, 1.02, f"Suspected patient sample IDs: {ids}", ha="left", va="center", fontsize=12, color="#8f1638", transform=ax.transAxes)

    table = ax.table(
        cellText=clipped.values,
        colLabels=clipped.columns,
        cellLoc="left",
        loc="upper center",
        bbox=[0, 0.02, 1, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.08, 1.45)

    n_cols = len(clipped.columns)
    for c in range(n_cols):
        head = table[(0, c)]
        head.set_facecolor("#1e8f4e")
        head.set_edgecolor("#16673a")
        head.get_text().set_color("white")
        head.get_text().set_weight("bold")

    for r in range(1, len(clipped) + 1):
        for c in range(n_cols):
            cell = table[(r, c)]
            cell.set_edgecolor("#cde8d5")
            cell.get_text().set_color("#b00020")
            if r == 1:
                cell.set_facecolor("#fdf1f4")

    footer = f"Generated: {generated_at}"
    if len(view) > max_rows:
        footer += f" | Showing first {max_rows} of {len(view)} suspected patient rows"
    ax.text(0.0, -0.02, footer, ha="left", va="top", fontsize=10.5, color="#666", transform=ax.transAxes)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


st.set_page_config(page_title="IEM Disease Prediction", layout="wide")
load_local_env()

st.title("IEM Disease Prediction Web App")
st.write("Upload one or more CSV files to get top-3 IEM disease predictions")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="comparisons/best_smote_balanced_lr.joblib")
    class_mapping_csv = st.text_input("Class mapping CSV", value="comparisons/class_mapping.csv")
    id_column = st.text_input(
        "ID column override (optional)",
        value="",
        help=(
            "Leave blank for auto-detect (recommended). "
            "Use this only if your sample ID column has a custom name."
        ),
    )
    default_webhook = get_discord_webhook()
    auto_send_discord = st.checkbox(
        "Auto-send suspected cases to Discord after prediction",
        value=False,
    )
    if default_webhook:
        if st.button("Send Test Ping to Discord"):
            try:
                send_discord_message(default_webhook, "IEM app test ping from Streamlit sidebar.")
                st.success("Test ping sent to Discord.")
            except Exception as exc:
                st.error(f"Test ping failed: {exc}")
    else:
        st.caption("Set DISCORD_WEBHOOK_URL in env/.env to enable Discord sending.")

uploaded_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
run_btn = st.button("Run Prediction", type="primary", disabled=(not uploaded_files))

if "keep_results_visible" not in st.session_state:
    st.session_state["keep_results_visible"] = False

if run_btn:
    st.session_state["keep_results_visible"] = True

if not uploaded_files:
    st.session_state["keep_results_visible"] = False

if (run_btn or st.session_state.get("keep_results_visible", False)) and uploaded_files:
    runs_dir = Path("comparisons") / "web_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_output_csv = runs_dir / f"predictions_combined_{ts}.csv"
    combined_suspected_html = runs_dir / f"suspected_report_combined_{ts}.html"
    combined_suspected_png = runs_dir / f"suspected_report_combined_{ts}.png"
    combined_html_zip = runs_dir / f"reports_{ts}.zip"

    try:
        result_frames = []
        html_reports: Dict[str, Path] = {}
        total_rows = 0

        for idx, up in enumerate(uploaded_files):
            input_path = runs_dir / f"input_{ts}_{idx}_{Path(up.name).stem}.csv"
            output_csv = runs_dir / f"predictions_{ts}_{idx}_{Path(up.name).stem}.csv"
            output_html = runs_dir / f"report_{ts}_{idx}_{Path(up.name).stem}.html"
            input_path.write_bytes(up.getvalue())

            result = cast(Dict[str, Any], run_inference(
                model_path=model_path,
                input_csv=str(input_path),
                output_csv=str(output_csv),
                id_column=id_column.strip() or None,
                class_mapping_csv=class_mapping_csv.strip() or None,
                html_report=str(output_html),
            ))

            df_part = cast(pd.DataFrame, result["results_df"]).copy()
            source_id_col = str(result.get("source_id_column", "")).strip()
            input_df = pd.read_csv(input_path)
            if source_id_col and source_id_col in input_df.columns:
                marker_info_df = build_marker_info_table(input_df, source_id_col)
                df_part = df_part.merge(marker_info_df, on="sample_id", how="left")
            else:
                df_part["top_1_marker"] = "-"
                df_part["top_1_marker_value"] = "-"

            df_part["top_1_marker"] = df_part["top_1_marker"].fillna("-").astype(str)
            df_part["top_1_marker_value"] = df_part["top_1_marker_value"].fillna("-").astype(str)
            df_part["source_file"] = up.name
            result_frames.append(df_part)
            html_reports[up.name] = output_html
            total_rows += int(result["rows_predicted"])

        if not result_frames:
            raise RuntimeError("No CSV files were processed.")

        df = pd.concat(result_frames, axis=0, ignore_index=True)
        id_col = "sample_id"

        df["is_flagged"] = ~df["top_1_disease"].astype(str).apply(is_normal_disease_name)
        df["is_control"] = df[id_col].astype(str).apply(is_control_or_internal_sample)
        df["priority"] = df["is_flagged"].map({True: "FLAG", False: "NORMAL"})

        flagged_df = df[df["is_flagged"]].copy()
        patient_flagged_df = get_patient_suspected_cases(df, id_col)
        normal_df = df[~df["is_flagged"]].copy()
        # Group order: flagged patients -> control/internal samples -> normal patients
        df["display_group"] = 2
        df.loc[df["is_control"], "display_group"] = 1
        df.loc[df["is_flagged"] & ~df["is_control"], "display_group"] = 0
        prioritized_df = df.sort_values(by=["display_group"], ascending=[True], kind="stable").reset_index(drop=True)
        prioritized_df.to_csv(combined_output_csv, index=False)

        st.success(
            f"Prediction complete for {len(uploaded_files)} file(s). "
            f"Total rows: {total_rows}, Sample ID column: {id_col}"
        )

        st.subheader("Priority Triage (Flagged Samples First)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", len(df))
        c2.metric("Flagged Patients (Disease)", len(patient_flagged_df))
        c3.metric("Normal/Other", len(normal_df))

        if not patient_flagged_df.empty:
            flagged_ids = ", ".join(patient_flagged_df[id_col].astype(str).tolist())
            st.error(f"Priority sample IDs for doctor review: {flagged_ids}")
        else:
            st.success("No flagged disease samples detected in top-1 prediction.")

        st.subheader("Top-3 Predictions (with Sample ID)")
        show_cols = [
            c
            for c in [
                id_col,
                "source_file",
                "priority",
                "top_1_disease",
                "top_1_probability",
                "top_1_marker",
                "top_1_marker_value",
                "top_2_disease",
                "top_2_probability",
                "top_3_disease",
                "top_3_probability",
            ]
            if c in prioritized_df.columns
        ]

        display_df = prioritized_df[show_cols].copy()
        display_df = display_df.rename(columns={id_col: "sample_id"})

        styled = display_df.style.apply(
            lambda row: [
                "color: #b00020; font-weight: 700;" if row.get("priority") == "FLAG" else "color: #0b7a3f;"
            ] * len(row),
            axis=1,
        )
        st.dataframe(styled, width="stretch")

        suspected_only = prioritized_df[prioritized_df["is_flagged"]].copy()
        suspected_cols = [
            c
            for c in [
                id_col,
            "source_file",
                "top_1_disease",
                "top_1_probability",
                "top_1_marker",
                "top_1_marker_value",
                "top_2_disease",
                "top_2_probability",
                "top_3_disease",
                "top_3_probability",
            ]
            if c in suspected_only.columns
        ]
        suspected_csv_bytes = suspected_only[suspected_cols].to_csv(index=False).encode("utf-8")
        suspected_html_path = build_suspected_html_report(prioritized_df, id_col, str(combined_suspected_html))
        suspected_png_path = None
        try:
            suspected_png_path = build_suspected_png_report(prioritized_df, id_col, str(combined_suspected_png))
        except Exception as exc:
            st.warning(f"Could not build PNG report, using HTML for Discord attachment instead: {exc}")

        st.subheader("Report Preview (HTML Format)")
        preview_options = list(html_reports.keys())
        preview_name = st.selectbox("Select file report to preview", options=preview_options)
        html_content = Path(html_reports[preview_name]).read_text(encoding="utf-8")
        components.html(html_content, height=900, scrolling=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_name, report_path in html_reports.items():
                zf.write(report_path, arcname=f"{Path(file_name).stem}_report.html")
        zip_buffer.seek(0)

        st.subheader("Downloads")
        st.download_button(
            label="Download Combined Predictions CSV",
            data=Path(combined_output_csv).read_bytes(),
            file_name=combined_output_csv.name,
            mime="text/csv",
        )
        st.download_button(
            label="Download All HTML Reports (ZIP)",
            data=zip_buffer.getvalue(),
            file_name=combined_html_zip.name,
            mime="application/zip",
        )
        st.download_button(
            label="Download Combined Suspected Cases CSV",
            data=suspected_csv_bytes,
            file_name=f"suspected_cases_combined_{ts}.csv",
            mime="text/csv",
        )
        st.download_button(
            label="Download Combined Suspected Cases HTML",
            data=Path(suspected_html_path).read_bytes(),
            file_name=Path(suspected_html_path).name,
            mime="text/html",
        )
        if suspected_png_path:
            st.download_button(
                label="Download Combined Suspected Cases PNG",
                data=Path(suspected_png_path).read_bytes(),
                file_name=Path(suspected_png_path).name,
                mime="image/png",
            )

        st.subheader("Discord")
        if run_btn and auto_send_discord and default_webhook:
            try:
                msg = build_discord_suspected_summary(prioritized_df, id_col)
                send_discord_message(default_webhook, msg)
                if suspected_png_path:
                    send_discord_file_attachment(
                        default_webhook,
                        suspected_png_path,
                        "Suspected-cases combined report attached (patient only).",
                        mime_type="image/png",
                    )
                    st.success("Auto-sent combined suspected-cases summary + PNG attachment to Discord.")
                else:
                    send_discord_file_attachment(
                        default_webhook,
                        suspected_html_path,
                        "Suspected-cases combined report attached (HTML fallback, patient only).",
                        mime_type="text/html",
                    )
                    st.success("Auto-sent combined suspected-cases summary + HTML fallback attachment to Discord.")
            except urlerror.URLError as exc:
                st.error(f"Discord network error during auto-send: {exc}")
            except Exception as exc:
                st.error(f"Discord auto-send failed: {exc}")

        send_btn = st.button(
            "Send Suspected Cases to Discord",
            disabled=(not default_webhook),
        )
        if send_btn:
            try:
                msg = build_discord_suspected_summary(prioritized_df, id_col)
                send_discord_message(default_webhook, msg)
                if suspected_png_path:
                    send_discord_file_attachment(
                        default_webhook,
                        suspected_png_path,
                        "Suspected-cases combined report attached (patient only).",
                        mime_type="image/png",
                    )
                    st.success("Sent combined suspected-cases summary + PNG attachment to Discord.")
                else:
                    send_discord_file_attachment(
                        default_webhook,
                        suspected_html_path,
                        "Suspected-cases combined report attached (HTML fallback, patient only).",
                        mime_type="text/html",
                    )
                    st.success("Sent combined suspected-cases summary + HTML fallback attachment to Discord.")
            except urlerror.URLError as exc:
                st.error(f"Discord network error: {exc}")
            except Exception as exc:
                st.error(f"Discord send failed: {exc}")

        st.info(
            "Files saved under comparisons/web_runs. "
            "You can also open the HTML report directly in your browser."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.caption("Local app for clinical-research workflow support.")
