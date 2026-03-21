import argparse
from datetime import datetime
import os
import sys
from typing import Any, Dict, List, Optional, cast

import joblib
import numpy as np
import pandas as pd


class LogTransformer:
    """Compatibility class for models saved from notebook pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_num = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").fillna(0)
        X_num = X_num.clip(lower=-0.999999)
        return np.log1p(X_num)

    def fit_transform(self, X, y=None):
        return self.transform(X)


def register_pickle_compat_classes() -> None:
    """Expose custom classes on __main__ for joblib/pickle compatibility."""
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "LogTransformer"):
        setattr(main_mod, "LogTransformer", LogTransformer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict disease probabilities for multiple rows using a trained joblib model."
    )
    parser.add_argument("--model-path", default="comparisons/best_smote_balanced_lr.joblib")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument(
        "--output-csv",
        default="comparisons/inference_predictions_with_probabilities.csv",
    )
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--class-mapping-csv", default=None)
    parser.add_argument(
        "--html-report",
        default="comparisons/inference_prediction_report.html",
    )
    return parser.parse_args()


def normalize_sample_id(value) -> str:
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        return text[2:-1]
    return text


def detect_id_column(df: pd.DataFrame, user_id_col: Optional[str]) -> str:
    if user_id_col is not None:
        if user_id_col not in df.columns:
            raise ValueError(f"Requested id-column '{user_id_col}' not found in input CSV")
        return user_id_col

    candidates = ["Barcode", "LabNumber", "ID", "id", "sample_id", "SampleID", "HN"]
    for col in candidates:
        if col in df.columns:
            return col

    df["row_id"] = np.arange(1, len(df) + 1)
    return "row_id"


def get_expected_features(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def load_class_mapping(mapping_csv: str) -> Dict[int, str]:
    mapping_df = pd.read_csv(mapping_csv)
    required = {"class_id", "class_name"}
    if not required.issubset(mapping_df.columns):
        raise ValueError(
            f"Class mapping CSV must contain columns {required}, got {list(mapping_df.columns)}"
        )
    return dict(zip(mapping_df["class_id"], mapping_df["class_name"]))


def resolve_disease_names(model, class_mapping_csv: Optional[str]) -> List[str]:
    class_ids = list(model.classes_)

    if class_mapping_csv is not None:
        mapping = load_class_mapping(class_mapping_csv)
        return [str(mapping.get(cid, f"class_{cid}")) for cid in class_ids]

    if all(isinstance(c, str) for c in class_ids):
        return [str(c) for c in class_ids]

    return [f"class_{cid}" for cid in class_ids]


def get_class_mapping_path(user_path: Optional[str]) -> Optional[str]:
    if user_path is not None:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"Class mapping file not found: {user_path}")
        return user_path

    default_path = "comparisons/class_mapping.csv"
    if os.path.exists(default_path):
        return default_path

    return None


def is_true_like(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def get_report_sample_mask(df: pd.DataFrame) -> pd.Series:
    if "For Answer" in df.columns:
        return df["For Answer"].apply(is_true_like)

    if "Barcode" in df.columns:
        barcode_text = df["Barcode"].astype(str).str.strip().str.upper()
        return barcode_text.ne("IS")

    return pd.Series([True] * len(df), index=df.index)


def parse_abnormal_markers(value) -> List[str]:
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if ":" in text:
        text = text.split(":", 1)[1]

    return [marker.strip() for marker in text.split(";") if marker.strip()]


def format_probability(value: float) -> str:
    return f"{value * 100:.2f}%"


def is_normal_disease_name(name: str) -> bool:
    text = str(name).strip().lower()
    return text == "normal/other unspecified disease" or text.startswith("normal")


def build_html_report(
    input_df: pd.DataFrame,
    results_df: pd.DataFrame,
    id_col: str,
    html_report_path: str,
) -> None:
    report_mask = get_report_sample_mask(input_df)
    filtered_input = input_df.loc[report_mask].reset_index(drop=True)
    filtered_results = results_df.loc[report_mask].reset_index(drop=True)

    report_order = sorted(
        range(len(filtered_results)),
        key=lambda i: is_normal_disease_name(filtered_results.iloc[i]["top_1_disease"]),
    )

    filtered_input = filtered_input.iloc[report_order].reset_index(drop=True)
    filtered_results = filtered_results.iloc[report_order].reset_index(drop=True)

    sample_label_col = "Barcode" if "Barcode" in filtered_input.columns else id_col

    suspect_sample_ids: List[str] = []
    sections = []
    for idx in range(len(filtered_results)):
        source_row = filtered_input.iloc[idx]
        pred_row = filtered_results.iloc[idx]

        if sample_label_col in source_row.index:
            sample_id_raw = source_row[sample_label_col]
        elif "sample_id" in pred_row.index:
            sample_id_raw = pred_row["sample_id"]
        else:
            sample_id_raw = pred_row.get(id_col, "-")

        sample_id = normalize_sample_id(sample_id_raw)
        top1_is_normal = is_normal_disease_name(pred_row["top_1_disease"])
        is_flagged_sample = not top1_is_normal
        if is_flagged_sample:
            suspect_sample_ids.append(sample_id)

        abnormal_markers = parse_abnormal_markers(source_row.get("Abnormals", ""))

        table_rows = []
        for rank in range(1, 4):
            disease = pred_row[f"top_{rank}_disease"]

            probability = format_probability(pred_row[f"top_{rank}_probability"])
            row_class = "normal-row" if top1_is_normal else "disease-row"

            marker = abnormal_markers[rank - 1] if len(abnormal_markers) >= rank else "-"
            marker_value = "-"
            if marker != "-" and marker in source_row.index:
                marker_value = source_row[marker]

            table_rows.append(
                f"""
                <tr class=\"{row_class}\">
                    <td class=\"disease\">{disease}</td>
                    <td class=\"prob\">{probability}</td>
                    <td>{marker}</td>
                    <td>{marker_value}</td>
                </tr>
                """
            )

        abnormal_note = ""
        if abnormal_markers:
            abnormal_note = (
                "<div class=\"sample-note\"><strong>Flagged markers:</strong> "
                + ", ".join(abnormal_markers)
                + "</div>"
            )

        sections.append(
            f"""
            <section class=\"sample-section\">
                <div class=\"sample-title {'flagged-sample-title' if is_flagged_sample else 'normal-sample-title'}\">Sample ID: {sample_id} {'[FLAG]' if is_flagged_sample else ''}</div>
                <table>
                    <thead>
                        <tr>
                            <th>Disease Pattern</th>
                            <th>Probability</th>
                            <th>Flagged Marker</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
                {abnormal_note}
            </section>
            """
        )

    if suspect_sample_ids:
        suspect_block = (
            "<div class=\"suspect-summary\"><strong>Suspected disease sample IDs (grouped):</strong> "
            + ", ".join(suspect_sample_ids)
            + "</div>"
        )
    else:
        suspect_block = (
            "<div class=\"suspect-summary normal-summary\"><strong>Suspected disease sample IDs (grouped):</strong> None</div>"
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <title>Disease Pattern Detection Report</title>
        <style>
            body {{
                font-family: Arial, Helvetica, sans-serif;
                margin: 24px 32px 56px;
                color: #222;
                background: #fff;
            }}
            .page-title {{
                background: #1e8f4e;
                color: #fff;
                text-align: center;
                font-size: 22px;
                font-weight: 700;
                padding: 12px 16px;
                border-radius: 2px;
                margin-bottom: 18px;
            }}
            .report-note {{
                font-size: 12px;
                color: #666;
                margin-bottom: 18px;
            }}
            .sample-section {{
                margin-bottom: 22px;
                page-break-inside: avoid;
            }}
            .sample-title {{
                font-size: 16px;
                font-weight: 700;
                margin-bottom: 8px;
            }}
            .flagged-sample-title {{
                color: #b00020;
            }}
            .normal-sample-title {{
                color: #0b7a3f;
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
            .suspect-summary.normal-summary {{
                border-color: #a6dcb7;
                background: #eefaf1;
                color: #18633a;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
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
                vertical-align: top;
                word-break: break-word;
            }}
            tbody tr.disease-row td {{
                color: #b00020;
            }}
            tbody tr.normal-row td {{
                color: #0b7a3f;
            }}
            tbody tr:nth-child(1) td {{
                font-weight: 700;
                background: #f2fbf5;
            }}
            .disease {{ width: 44%; }}
            .prob {{ width: 14%; text-align: center; }}
            .sample-note {{
                margin-top: 8px;
                font-size: 12px;
                color: #666;
            }}
            .footer {{
                margin-top: 40px;
                text-align: center;
                color: #8a8a8a;
                font-size: 12px;
                font-style: italic;
            }}
            @media print {{
                body {{ margin: 16px 18px 32px; }}
                .sample-section {{ page-break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <div class=\"page-title\">Disease Pattern Detection Report</div>
        <div class=\"report-note\">
            Report uses model top-3 predicted diseases per sample. Marker and value columns show flagged sample markers when available.
            Disease-specific match counts and MoM are not generated by this model.
        </div>
        {suspect_block}
        {''.join(sections)}
        <div class=\"footer\">Generated: {generated_at} | Automated Disease Prediction Report</div>
    </body>
    </html>
    """

    os.makedirs(os.path.dirname(html_report_path) or ".", exist_ok=True)
    with open(html_report_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(html)


def run_inference(
    model_path: str,
    input_csv: str,
    output_csv: str,
    id_column: Optional[str] = None,
    class_mapping_csv: Optional[str] = None,
    html_report: Optional[str] = None,
) -> Dict[str, Any]:
    register_pickle_compat_classes()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    model = joblib.load(model_path)
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Loaded model does not support predict_proba()")

    df = pd.read_csv(input_csv)
    id_col = detect_id_column(df, id_column)

    non_feature_cols = ["EnzymeDefect", "EnzymeDefect_enc", id_col]
    X = df.drop(columns=[c for c in non_feature_cols if c in df.columns]).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    expected_features = get_expected_features(model)
    if expected_features is not None:
        X = X.reindex(columns=expected_features, fill_value=0)

    probs = model.predict_proba(X)
    class_ids = list(model.classes_)
    class_mapping_path = get_class_mapping_path(class_mapping_csv)
    disease_names = resolve_disease_names(model, class_mapping_path)

    if class_mapping_path is None and not all(isinstance(c, str) for c in class_ids):
        template_path = "comparisons/class_mapping_template.csv"
        template_df = pd.DataFrame(
            {
                "class_id": class_ids,
                "class_name": [f"class_{cid}" for cid in class_ids],
            }
        )
        template_df.to_csv(template_path, index=False)

    prob_cols = [f"prob_{name}" for name in disease_names]
    proba_df = pd.DataFrame(probs, columns=prob_cols)

    top_k = min(3, len(class_ids))
    top_indices = np.argsort(-probs, axis=1)[:, :top_k]

    top_predictions = {}
    for rank in range(top_k):
        rank_indices = top_indices[:, rank]
        top_predictions[f"top_{rank + 1}_class_id"] = [class_ids[i] for i in rank_indices]
        top_predictions[f"top_{rank + 1}_disease"] = [disease_names[i] for i in rank_indices]
        top_predictions[f"top_{rank + 1}_probability"] = probs[np.arange(len(probs)), rank_indices]

    sample_id_series = df[id_col].map(normalize_sample_id)

    results_df = pd.concat(
        [
            pd.DataFrame({"sample_id": sample_id_series}).reset_index(drop=True),
            pd.DataFrame(top_predictions),
            proba_df.reset_index(drop=True),
        ],
        axis=1,
    )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    if html_report:
        build_html_report(df, results_df, id_col, html_report)

    return {
        "rows_predicted": len(results_df),
        "id_column": "sample_id",
        "source_id_column": id_col,
        "class_mapping_path": class_mapping_path,
        "output_csv": output_csv,
        "html_report": html_report,
        "results_df": results_df,
    }


def main() -> None:
    args = parse_args()

    result = run_inference(
        model_path=args.model_path,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        id_column=args.id_column,
        class_mapping_csv=args.class_mapping_csv,
        html_report=args.html_report,
    )

    if result["class_mapping_path"] is not None:
        print(f"Using class mapping: {result['class_mapping_path']}")

    print(f"Rows predicted: {result['rows_predicted']}")
    print(f"ID column used: {result['id_column']}")
    print(f"Saved predictions: {result['output_csv']}")
    print(f"Saved HTML report: {result['html_report']}")
    print("Preview:")
    preview_df = cast(pd.DataFrame, result["results_df"])
    print(preview_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
