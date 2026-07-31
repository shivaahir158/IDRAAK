"""Streamlit human evaluation interface for IDRAAK drift detection.

Run with:
    streamlit run src/idraak/ui/app.py

Allows evaluators to review requirement pairs, see model predictions,
rate drift detection quality, and export ratings.
"""

from __future__ import annotations

import csv
import io
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from idraak.schemas.dataset import DatasetEntry, PerturbationRecord
from idraak.schemas.drift import DriftType, DriftSeverity
from idraak.extraction.deterministic import DeterministicExtractor
from idraak.drift.comparator import SRRComparator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/idraak/ui -> project root
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REQUIREMENTS_PATH = DATA_DIR / "requirements.jsonl"
PERTURBATIONS_PATH = DATA_DIR / "perturbations.jsonl"
EVAL_RESULTS_PATH = REPORTS_DIR / "evaluation_results.jsonl"

DRIFT_TYPE_OPTIONS = ["none"] + [dt.value for dt in DriftType]
SEVERITY_OPTIONS = [ds.value for ds in DriftSeverity]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@st.cache_data
def load_requirements() -> dict[str, DatasetEntry]:
    """Load base requirements keyed by requirement_id."""
    raw = _load_jsonl(REQUIREMENTS_PATH)
    return {r["requirement_id"]: DatasetEntry(**r) for r in raw}


@st.cache_data
def load_perturbations() -> list[PerturbationRecord]:
    """Load perturbation records."""
    raw = _load_jsonl(PERTURBATIONS_PATH)
    return [PerturbationRecord(**r) for r in raw]


@st.cache_data
def load_eval_results() -> dict[str, dict[str, Any]]:
    """Load evaluation results keyed by requirement_id."""
    raw = _load_jsonl(EVAL_RESULTS_PATH)
    return {r["requirement_id"]: r for r in raw}


# ---------------------------------------------------------------------------
# SRR extraction & comparison helpers
# ---------------------------------------------------------------------------

_extractor = DeterministicExtractor()
_comparator = SRRComparator()


def extract_and_compare(original_text: str, perturbed_text: str, req_id: str = ""):
    """Extract SRRs from both texts and return structured comparison."""
    srr_orig = _extractor.extract(original_text, requirement_id=f"{req_id}_orig")
    srr_pert = _extractor.extract(perturbed_text, requirement_id=f"{req_id}_pert")
    diffs = _comparator.compare(srr_orig, srr_pert)
    return srr_orig, srr_pert, diffs


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------


def _init_session_state():
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
    if "evaluator_id" not in st.session_state:
        st.session_state.evaluator_id = ""
    if "blind_mode" not in st.session_state:
        st.session_state.blind_mode = False
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _colored_severity(severity: str) -> str:
    """Return severity with color-coded emoji indicator."""
    mapping = {
        "none": "white[none]",
        "low": "green[low]",
        "medium": "orange[medium]",
        "high": "red[high]",
        "critical": "red[CRITICAL]",
    }
    return mapping.get(severity, severity)


def _render_text_diff(original: str, perturbed: str):
    """Render original and perturbed texts side by side with highlighting."""
    orig_words = original.split()
    pert_words = perturbed.split()

    # Simple word-level diff highlighting
    highlighted_orig = []
    highlighted_pert = []

    max_len = max(len(orig_words), len(pert_words))
    for i in range(max_len):
        ow = orig_words[i] if i < len(orig_words) else ""
        pw = pert_words[i] if i < len(pert_words) else ""
        if ow.lower() != pw.lower():
            highlighted_orig.append(f"**:red[{ow}]**" if ow else "")
            highlighted_pert.append(f"**:red[{pw}]**" if pw else "")
        else:
            highlighted_orig.append(ow)
            highlighted_pert.append(pw)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original:**")
        st.markdown(" ".join(highlighted_orig))
    with col2:
        st.markdown("**Perturbed (back-translated):**")
        st.markdown(" ".join(highlighted_pert))


def _render_field_diffs(diffs):
    """Render field-level differences as a structured table."""
    if not diffs:
        st.info("No field-level differences detected by deterministic comparator.")
        return

    for i, diff in enumerate(diffs):
        severity = diff.severity.value if diff.severity else "none"
        drift_type = diff.drift_type.value if diff.drift_type else "n/a"

        severity_colors = {
            "none": "gray", "low": "green", "medium": "orange",
            "high": "red", "critical": "red",
        }
        color = severity_colors.get(severity, "gray")

        with st.expander(
            f"Diff #{i+1}: `{diff.field}` | {drift_type} | :{color}[{severity}] | conf={diff.confidence:.2f}",
            expanded=(severity in ("high", "critical")),
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Original value:** `{diff.original}`")
            with c2:
                st.markdown(f"**Candidate value:** `{diff.candidate}`")
            if diff.explanation:
                st.markdown(f"**Explanation:** {diff.explanation}")
            st.markdown(f"**Difference type:** {diff.difference_type}")


def _render_srr_comparison(srr_orig, srr_pert):
    """Render side-by-side SRR structured comparison."""
    fields = [
        ("Modality", lambda s: s.modality.value if s.modality else "n/a"),
        ("Polarity", lambda s: s.polarity.value if s.polarity else "n/a"),
        ("Actor", lambda s: s.actor or "n/a"),
        ("Action", lambda s: s.action or "n/a"),
        ("Object", lambda s: s.object or "n/a"),
        ("Conditions", lambda s: "; ".join(c.raw_text for c in s.conditions) or "n/a"),
        ("Temporal", lambda s: "; ".join(t.raw_text for t in s.temporal_constraints) or "n/a"),
        ("Numerical", lambda s: "; ".join(n.raw_text for n in s.numerical_constraints) or "n/a"),
        ("Ordering", lambda s: "; ".join(o.raw_text for o in s.ordering_constraints) or "n/a"),
        ("Exceptions", lambda s: "; ".join(e.raw_text for e in s.exceptions) or "n/a"),
        ("Units", lambda s: ", ".join(s.units) or "n/a"),
    ]

    rows = []
    for label, fn in fields:
        ov = fn(srr_orig)
        pv = fn(srr_pert)
        match = "Yes" if ov == pv else "**No**"
        rows.append({"Field": label, "Original SRR": ov, "Perturbed SRR": pv, "Match": match})

    st.table(rows)


# ---------------------------------------------------------------------------
# Rating form
# ---------------------------------------------------------------------------


def _render_rating_form(pair_id: str, existing_rating: dict[str, Any] | None):
    """Render the human evaluation rating form and return the rating dict."""
    st.subheader("Human Evaluation")

    defaults = existing_rating or {}

    with st.form(key=f"rating_form_{pair_id}"):
        col1, col2 = st.columns(2)

        with col1:
            drift_present = st.radio(
                "Is semantic drift present?",
                options=["Yes", "No", "Uncertain"],
                index=["Yes", "No", "Uncertain"].index(defaults.get("drift_present", "Uncertain")),
                key=f"drift_present_{pair_id}",
            )

            drift_type = st.selectbox(
                "What type of drift?",
                options=DRIFT_TYPE_OPTIONS,
                index=DRIFT_TYPE_OPTIONS.index(defaults.get("drift_type", "none")),
                key=f"drift_type_{pair_id}",
            )

            severity = st.selectbox(
                "How severe is the drift?",
                options=SEVERITY_OPTIONS,
                index=SEVERITY_OPTIONS.index(defaults.get("severity", "none")),
                key=f"severity_{pair_id}",
            )

        with col2:
            explanation_correct = st.radio(
                "Is the model explanation correct?",
                options=["Yes", "Partially", "No", "N/A"],
                index=["Yes", "Partially", "No", "N/A"].index(
                    defaults.get("explanation_correct", "N/A")
                ),
                key=f"explanation_correct_{pair_id}",
            )

            evidence_sufficient = st.radio(
                "Is the evidence sufficient?",
                options=["Yes", "Partially", "No", "N/A"],
                index=["Yes", "Partially", "No", "N/A"].index(
                    defaults.get("evidence_sufficient", "N/A")
                ),
                key=f"evidence_sufficient_{pair_id}",
            )

            human_review_required = st.radio(
                "Would human review be required in production?",
                options=["Yes", "No"],
                index=["Yes", "No"].index(defaults.get("human_review_required", "No")),
                key=f"human_review_{pair_id}",
            )

        notes = st.text_area(
            "Additional notes (optional)",
            value=defaults.get("notes", ""),
            key=f"notes_{pair_id}",
        )

        submitted = st.form_submit_button("Save Rating", type="primary")

    if submitted:
        rating = {
            "pair_id": pair_id,
            "evaluator_id": st.session_state.evaluator_id,
            "session_id": st.session_state.session_id,
            "timestamp": datetime.now().isoformat(),
            "drift_present": drift_present,
            "drift_type": drift_type,
            "severity": severity,
            "explanation_correct": explanation_correct,
            "evidence_sufficient": evidence_sufficient,
            "human_review_required": human_review_required,
            "notes": notes,
        }
        st.session_state.ratings[pair_id] = rating
        st.success(f"Rating saved for {pair_id}.")
        return rating

    return existing_rating


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_csv(ratings: dict[str, dict[str, Any]]) -> str:
    """Export ratings to CSV string."""
    if not ratings:
        return ""
    fieldnames = [
        "pair_id", "evaluator_id", "session_id", "timestamp",
        "drift_present", "drift_type", "severity",
        "explanation_correct", "evidence_sufficient",
        "human_review_required", "notes",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in ratings.values():
        writer.writerow(r)
    return output.getvalue()


def _export_jsonl(ratings: dict[str, dict[str, Any]]) -> str:
    """Export ratings to JSONL string."""
    lines = [json.dumps(r, ensure_ascii=False) for r in ratings.values()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="IDRAAK - Human Evaluation",
        page_icon="[D]",
        layout="wide",
    )

    _init_session_state()

    st.title("IDRAAK - Human Evaluation Interface")
    st.caption("Semantic drift detection evaluation for translated requirements")

    # --- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.header("Settings")

        st.session_state.evaluator_id = st.text_input(
            "Evaluator ID",
            value=st.session_state.evaluator_id,
            placeholder="e.g., evaluator-01",
        )

        st.session_state.blind_mode = st.toggle(
            "Blind evaluation (hide model identity & predictions)",
            value=st.session_state.blind_mode,
        )

        st.divider()
        st.header("Progress")
        total_pairs = len(load_perturbations())
        rated_count = len(st.session_state.ratings)
        if total_pairs > 0:
            st.progress(rated_count / total_pairs)
        st.write(f"{rated_count} / {total_pairs} rated")

        st.divider()
        st.header("Export Ratings")

        if st.session_state.ratings:
            csv_data = _export_csv(st.session_state.ratings)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"idraak_ratings_{st.session_state.session_id}.csv",
                mime="text/csv",
            )
            jsonl_data = _export_jsonl(st.session_state.ratings)
            st.download_button(
                label="Download JSONL",
                data=jsonl_data,
                file_name=f"idraak_ratings_{st.session_state.session_id}.jsonl",
                mime="application/jsonl",
            )
        else:
            st.info("No ratings to export yet.")

        st.divider()
        st.header("Data Paths")
        st.caption(f"Requirements: {REQUIREMENTS_PATH}")
        st.caption(f"Perturbations: {PERTURBATIONS_PATH}")
        st.caption(f"Eval results: {EVAL_RESULTS_PATH}")

    # --- Load data ---------------------------------------------------------
    requirements = load_requirements()
    perturbations = load_perturbations()
    eval_results = load_eval_results()

    if not requirements:
        st.error(f"No requirements found at {REQUIREMENTS_PATH}")
        return
    if not perturbations:
        st.error(f"No perturbations found at {PERTURBATIONS_PATH}")
        return

    # --- Filter controls ---------------------------------------------------
    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        domain_options = sorted({r.domain for r in requirements.values()})
        selected_domains = st.multiselect(
            "Filter by domain", options=domain_options, default=[]
        )

    with filter_col2:
        drift_type_filter = st.multiselect(
            "Filter by drift type (gold label)",
            options=["none"] + [dt.value for dt in DriftType],
            default=[],
        )

    with filter_col3:
        rating_filter = st.selectbox(
            "Rating status",
            options=["All", "Unrated only", "Rated only"],
            index=0,
        )

    # Apply filters
    filtered = perturbations
    if selected_domains:
        base_ids_in_domain = {
            rid for rid, r in requirements.items() if r.domain in selected_domains
        }
        filtered = [p for p in filtered if p.base_requirement_id in base_ids_in_domain]
    if drift_type_filter:
        filtered = [
            p for p in filtered
            if (p.drift_type or "none") in drift_type_filter
        ]
    if rating_filter == "Unrated only":
        filtered = [p for p in filtered if p.requirement_id not in st.session_state.ratings]
    elif rating_filter == "Rated only":
        filtered = [p for p in filtered if p.requirement_id in st.session_state.ratings]

    if not filtered:
        st.warning("No pairs match the current filters.")
        return

    # --- Navigation --------------------------------------------------------
    st.divider()
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 3, 1, 1])

    with nav_col1:
        if st.button("Previous", disabled=st.session_state.current_index <= 0):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()

    with nav_col2:
        st.session_state.current_index = st.slider(
            "Pair index",
            min_value=0,
            max_value=len(filtered) - 1,
            value=min(st.session_state.current_index, len(filtered) - 1),
            label_visibility="collapsed",
        )

    with nav_col3:
        if st.button("Next", disabled=st.session_state.current_index >= len(filtered) - 1):
            st.session_state.current_index = min(
                len(filtered) - 1, st.session_state.current_index + 1
            )
            st.rerun()

    with nav_col4:
        jump_to = st.number_input(
            "Go to #", min_value=1, max_value=len(filtered),
            value=st.session_state.current_index + 1,
            label_visibility="collapsed",
        )
        if jump_to - 1 != st.session_state.current_index:
            st.session_state.current_index = jump_to - 1
            st.rerun()

    idx = st.session_state.current_index
    pair = filtered[idx]
    base_req = requirements.get(pair.base_requirement_id)
    eval_result = eval_results.get(pair.requirement_id)

    st.caption(f"Pair {idx + 1} of {len(filtered)} | ID: `{pair.requirement_id}`")
    if pair.requirement_id in st.session_state.ratings:
        st.success("This pair has been rated.")

    # --- Section 1: Requirement texts --------------------------------------
    st.divider()
    st.subheader("1. Requirement Texts")

    if base_req:
        st.markdown(f"**Original requirement:** {base_req.original_text}")
        st.markdown(f"**Domain:** `{base_req.domain}` | **Category:** `{base_req.category}` | **Difficulty:** `{base_req.difficulty}`")
    else:
        st.warning(f"Base requirement {pair.base_requirement_id} not found.")

    st.markdown(f"**Translated / perturbed requirement:** {pair.perturbed_text}")

    if pair.description:
        if not st.session_state.blind_mode:
            st.markdown(f"**Perturbation description:** {pair.description}")

    # Text diff highlighting
    if base_req:
        with st.expander("Show word-level diff", expanded=False):
            _render_text_diff(base_req.original_text, pair.perturbed_text)

    # --- Section 2: Model predictions (hidden in blind mode) ---------------
    st.divider()
    st.subheader("2. Predicted Drift Analysis")

    if st.session_state.blind_mode:
        st.info("Blind mode enabled: model predictions are hidden. Rate based on the texts alone.")
    else:
        # Show gold labels
        gold_col1, gold_col2, gold_col3 = st.columns(3)
        with gold_col1:
            drift_label = "DRIFT" if pair.drift_label == 1 else "NO DRIFT"
            st.metric("Gold Label", drift_label)
        with gold_col2:
            st.metric("Gold Drift Type", pair.drift_type or "none")
        with gold_col3:
            st.metric("Gold Severity", pair.severity or "none")

        # Show evaluation results if available
        if eval_result:
            st.markdown("---")
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            with pred_col1:
                pred_label = "DRIFT" if eval_result.get("pred_label", 0) == 1 else "NO DRIFT"
                st.metric("Predicted Label", pred_label)
            with pred_col2:
                st.metric("Confidence", f"{eval_result.get('confidence', 0):.2f}")
            with pred_col3:
                st.metric("Field Diffs Found", eval_result.get("n_diffs", 0))

            correct = eval_result.get("pred_label") == eval_result.get("gold_label")
            if correct:
                st.success("Prediction matches gold label.")
            else:
                st.error("Prediction does NOT match gold label.")

        # Deterministic SRR extraction and comparison
        if base_req:
            with st.expander("Structured SRR Comparison", expanded=True):
                srr_orig, srr_pert, diffs = extract_and_compare(
                    base_req.original_text, pair.perturbed_text, pair.requirement_id
                )
                _render_srr_comparison(srr_orig, srr_pert)

            with st.expander(f"Field-Level Differences ({len(diffs)} found)", expanded=bool(diffs)):
                _render_field_diffs(diffs)

            # Evidence highlight
            if diffs:
                with st.expander("Evidence Summary"):
                    for d in diffs:
                        if d.explanation:
                            st.markdown(f"- **{d.field}**: {d.explanation}")
                        if d.evidence_original or d.evidence_candidate:
                            st.markdown(
                                f"  - Original evidence: `{d.evidence_original}` | "
                                f"Candidate evidence: `{d.evidence_candidate}`"
                            )

    # --- Section 3: Human rating form --------------------------------------
    st.divider()
    existing = st.session_state.ratings.get(pair.requirement_id)
    _render_rating_form(pair.requirement_id, existing)

    # --- Section 4: Ratings summary ----------------------------------------
    st.divider()
    with st.expander(f"All Ratings Summary ({len(st.session_state.ratings)} total)"):
        if st.session_state.ratings:
            summary_rows = []
            for pid, r in st.session_state.ratings.items():
                summary_rows.append({
                    "Pair ID": pid,
                    "Drift?": r["drift_present"],
                    "Type": r["drift_type"],
                    "Severity": r["severity"],
                    "Explanation OK?": r["explanation_correct"],
                    "Evidence OK?": r["evidence_sufficient"],
                    "Needs Review?": r["human_review_required"],
                })
            st.dataframe(summary_rows, use_container_width=True)
        else:
            st.info("No ratings recorded yet.")


if __name__ == "__main__":
    main()
