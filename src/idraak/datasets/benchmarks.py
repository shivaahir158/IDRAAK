"""Loader for external benchmarks (PAWSX, XNLI) mapped to drift detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from idraak.schemas.dataset import PerturbationRecord
from idraak.utils.logging import get_logger

logger = get_logger("benchmarks")


def load_pawsx(
    languages: list[str] | None = None,
    split: str = "test",
    max_per_language: int = 0,
    cache_dir: str = "data/benchmarks",
) -> list[PerturbationRecord]:
    """Load PAWSX benchmark as drift detection pairs.

    PAWSX has adversarial paraphrase pairs. We map:
      label=1 (paraphrase) -> drift_label=0 (no drift / semantically equivalent)
      label=0 (not paraphrase) -> drift_label=1 (drift / meaning changed)

    Args:
        languages: Language codes to load (default: all 7)
        split: Dataset split (test/train/validation)
        max_per_language: Max examples per language (0=all)
        cache_dir: Where to cache the converted JSONL files
    """
    from datasets import load_dataset

    if languages is None:
        languages = ["en", "de", "es", "fr", "ja", "ko", "zh"]

    cache_path = Path(cache_dir) / "pawsx"
    cache_path.mkdir(parents=True, exist_ok=True)

    all_records: list[PerturbationRecord] = []

    for lang in languages:
        jsonl_path = cache_path / f"pawsx_{lang}_{split}.jsonl"

        if jsonl_path.exists():
            logger.info(f"Loading cached PAWSX {lang}/{split} from {jsonl_path}")
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        all_records.append(PerturbationRecord.model_validate_json(line))
            if max_per_language > 0:
                lang_records = [r for r in all_records if r.requirement_id.startswith(f"PAWSX-{lang}")]
                if len(lang_records) > max_per_language:
                    ids_to_keep = {r.requirement_id for r in lang_records[:max_per_language]}
                    all_records = [r for r in all_records if not r.requirement_id.startswith(f"PAWSX-{lang}") or r.requirement_id in ids_to_keep]
            continue

        logger.info(f"Downloading PAWSX {lang}/{split}...")
        ds = load_dataset("google-research-datasets/paws-x", lang, split=split)

        records = []
        for i, row in enumerate(ds):
            if max_per_language > 0 and i >= max_per_language:
                break

            # PAWSX: label=1 means paraphrase (no drift), label=0 means not paraphrase (drift)
            drift_label = 0 if row["label"] == 1 else 1
            drift_type = None if drift_label == 0 else "semantic_divergence"

            rec = PerturbationRecord(
                requirement_id=f"PAWSX-{lang}-{split}-{row['id']:05d}",
                base_requirement_id=f"PAWSX-{lang}-{split}-{row['id']:05d}-base",
                original_text=row["sentence1"],
                perturbed_text=row["sentence2"],
                drift_label=drift_label,
                drift_type=drift_type,
                source_language=lang,
                target_language=lang,
                metadata={
                    "benchmark": "pawsx",
                    "language": lang,
                    "original_label": row["label"],
                },
            )
            records.append(rec)

        # Cache to disk
        with open(jsonl_path, "w") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

        logger.info(f"  PAWSX {lang}: {len(records)} pairs (drift={sum(1 for r in records if r.drift_label==1)}, no-drift={sum(1 for r in records if r.drift_label==0)})")
        all_records.extend(records)

    return all_records


def load_xnli(
    languages: list[str] | None = None,
    split: str = "test",
    max_per_language: int = 0,
    include_neutral: bool = False,
    cache_dir: str = "data/benchmarks",
) -> list[PerturbationRecord]:
    """Load XNLI benchmark as drift detection pairs.

    XNLI has premise-hypothesis pairs with NLI labels. We map:
      entailment (0) -> drift_label=0 (no drift / meaning preserved)
      contradiction (2) -> drift_label=1 (drift / meaning changed)
      neutral (1) -> excluded by default, or drift_label=1 if include_neutral=True

    Uses English premise as original, cross-lingual hypothesis as candidate.

    Args:
        languages: Language codes (default: en, hi, ur, ar, zh, es, fr, de, tr)
        split: Dataset split (test/validation)
        max_per_language: Max examples per language (0=all)
        include_neutral: Whether to include neutral pairs as drift
        cache_dir: Where to cache converted JSONL files
    """
    from datasets import load_dataset

    if languages is None:
        languages = ["en", "hi", "ur", "ar", "zh", "es", "fr", "de", "tr"]

    cache_path = Path(cache_dir) / "xnli"
    cache_path.mkdir(parents=True, exist_ok=True)

    neutral_tag = "_with_neutral" if include_neutral else ""
    jsonl_path = cache_path / f"xnli_{split}{neutral_tag}.jsonl"

    if jsonl_path.exists():
        logger.info(f"Loading cached XNLI {split} from {jsonl_path}")
        all_records = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    all_records.append(PerturbationRecord.model_validate_json(line))
        # Filter to requested languages
        all_records = [r for r in all_records if r.target_language in languages]
        if max_per_language > 0:
            filtered = []
            lang_counts: dict[str, int] = {}
            for r in all_records:
                lang_counts.setdefault(r.target_language, 0)
                if lang_counts[r.target_language] < max_per_language:
                    filtered.append(r)
                    lang_counts[r.target_language] += 1
            return filtered
        return all_records

    logger.info(f"Downloading XNLI {split}...")
    ds = load_dataset("facebook/xnli", "all_languages", split=split)

    # XNLI label mapping
    LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

    all_records = []
    for idx, row in enumerate(ds):
        label = row["label"]

        # Skip neutral unless requested
        if label == 1 and not include_neutral:
            continue

        # entailment=no drift, contradiction/neutral=drift
        drift_label = 0 if label == 0 else 1
        drift_type = None if drift_label == 0 else "semantic_divergence"

        # English premise
        en_premise = row["premise"]["en"]

        # Get hypothesis translations
        hyp_langs = row["hypothesis"]["language"]
        hyp_translations = row["hypothesis"]["translation"]

        for lang_idx, lang in enumerate(hyp_langs):
            if lang not in languages:
                continue

            hypothesis = hyp_translations[lang_idx]

            rec = PerturbationRecord(
                requirement_id=f"XNLI-{lang}-{split}-{idx:05d}",
                base_requirement_id=f"XNLI-en-{split}-{idx:05d}-base",
                original_text=en_premise,
                perturbed_text=hypothesis,
                drift_label=drift_label,
                drift_type=drift_type,
                source_language="en",
                target_language=lang,
                metadata={
                    "benchmark": "xnli",
                    "nli_label": LABEL_MAP[label],
                    "language": lang,
                },
            )
            all_records.append(rec)

    # Cache to disk
    with open(jsonl_path, "w") as f:
        for rec in all_records:
            f.write(rec.model_dump_json() + "\n")

    logger.info(f"  XNLI total: {len(all_records)} pairs across {len(set(r.target_language for r in all_records))} languages")

    # Apply filters
    all_records = [r for r in all_records if r.target_language in languages]
    if max_per_language > 0:
        filtered = []
        lang_counts: dict[str, int] = {}
        for r in all_records:
            lang_counts.setdefault(r.target_language, 0)
            if lang_counts[r.target_language] < max_per_language:
                filtered.append(r)
                lang_counts[r.target_language] += 1
        return filtered

    return all_records
