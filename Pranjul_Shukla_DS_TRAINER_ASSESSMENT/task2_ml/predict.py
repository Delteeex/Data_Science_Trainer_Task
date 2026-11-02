"""Apply the saved FinTrust model to a raw application CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cleaning import SCENARIO_COLUMN, clean_inference_data
from src.features import predictor_frame
from src.phase2e_2j import validate_prediction_frame


def predict_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path | None = None,
) -> Path:
    """Clean raw applications consistently and save thresholded predictions."""
    project_dir = Path(__file__).resolve().parent
    artifact_path = (
        Path(model_path) if model_path is not None else project_dir / "models" / "final_model.joblib"
    )
    raw = pd.read_csv(input_path, low_memory=False)
    if "applicant_id" not in raw:
        raise ValueError("Input CSV must contain applicant_id.")
    if raw["applicant_id"].isna().any():
        raise ValueError("Input CSV contains missing applicant_id values.")
    cleaned = clean_inference_data(
        raw,
        allow_scenario_column=SCENARIO_COLUMN in raw.columns,
    )
    features = predictor_frame(cleaned)
    artifact = joblib.load(artifact_path)
    expected = artifact["input_features"]
    missing = sorted(set(expected) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"Input feature mismatch; missing={missing}, unexpected={unexpected}."
        )
    features = features.loc[:, expected]
    probabilities = np.asarray(
        artifact["pipeline"].predict_proba(features)[:, 1], dtype=float
    )
    threshold = float(artifact["threshold"])
    result = pd.DataFrame(
        {
            "applicant_id": raw["applicant_id"].astype(str).to_numpy(),
            "default_probability": probabilities,
            "default_pred": (probabilities >= threshold).astype("int64"),
        }
    )
    validate_prediction_frame(
        result, raw["applicant_id"], expected_rows=len(raw)
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, dest="input_csv")
    parser.add_argument("--output", required=True, type=Path, dest="output_csv")
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    result = predict_csv(args.input_csv, args.output_csv, model_path=args.model)
    print(f"Predictions saved to {result}")


if __name__ == "__main__":
    main()
