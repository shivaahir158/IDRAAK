"""Integration tests for workflows."""

import pytest
from idraak.providers.mock import MockLLMProvider, MockTranslationProvider
from idraak.workflows.direct_judge import DirectJudgeWorkflow
from idraak.workflows.full_idraak import FullIDRAAKWorkflow
from idraak.workflows.structured_single import StructuredSingleWorkflow


class TestDirectJudgeWorkflow:
    def test_mock_no_provider(self):
        wf = DirectJudgeWorkflow(llm_provider=None)
        result = wf.run(
            "The system shall respond within 5 ms.",
            "The system shall respond within 50 ms.",
            target_language="en",
            requirement_id="REQ-0001",
        )
        assert result.workflow == "direct_judge"
        assert result.requirement_id == "REQ-0001"

    def test_with_mock_llm(self):
        wf = DirectJudgeWorkflow(llm_provider=MockLLMProvider())
        result = wf.run(
            "The system shall respond within 5 ms.",
            "The system shall respond within 50 ms.",
            target_language="en",
        )
        assert result.workflow == "direct_judge"


class TestStructuredSingleWorkflow:
    def test_identical_texts(self):
        wf = StructuredSingleWorkflow(llm_provider=None)
        text = "The controller shall assert ready within 3 clock cycles."
        result = wf.run(text, text, requirement_id="REQ-0001")
        assert result.workflow == "structured_single"
        # Identical text should have no drift
        assert not result.drift_detected

    def test_different_modality(self):
        wf = StructuredSingleWorkflow(llm_provider=None)
        orig = "The system shall process the request."
        cand = "The system should process the request."
        result = wf.run(orig, cand, requirement_id="REQ-0002")
        assert result.drift_detected
        assert any("modality" in str(dt).lower() for dt in result.drift_types)

    def test_numerical_change(self):
        wf = StructuredSingleWorkflow(llm_provider=None)
        orig = "The latency shall not exceed 10 ms."
        cand = "The latency shall not exceed 100 ms."
        result = wf.run(orig, cand, requirement_id="REQ-0003")
        assert result.drift_detected


class TestFullIDRAAKWorkflow:
    def test_with_mock_providers(self):
        mock_trans = MockTranslationProvider()
        mock_llm = MockLLMProvider()
        wf = FullIDRAAKWorkflow(
            translation_provider=mock_trans,
            llm_provider=mock_llm,
        )
        result = wf.run(
            original_text="The controller shall assert ready within 3 clock cycles.",
            candidate_text="The controller should assert ready within 30 clock cycles.",
            requirement_id="REQ-0001",
        )
        assert result.workflow == "full_idraak"
        assert result.requirement_id == "REQ-0001"

    def test_without_candidate(self):
        mock_trans = MockTranslationProvider()
        wf = FullIDRAAKWorkflow(translation_provider=mock_trans)
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            source_language="en",
            target_language="hi",
            requirement_id="REQ-0002",
        )
        assert result.workflow == "full_idraak"

    def test_deterministic_only(self):
        wf = FullIDRAAKWorkflow()
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system may respond within 50 ms.",
            requirement_id="REQ-0003",
        )
        assert result.workflow == "full_idraak"
        assert result.drift_detected


class TestEndToEndPipeline:
    """Full pipeline: generate -> perturb -> extract -> compare -> evaluate."""

    def test_full_pipeline(self):
        from idraak.datasets.generator import DatasetGenerator
        from idraak.evaluation.metrics import ClassificationMetrics
        from idraak.perturbations.engine import PerturbationEngine

        gen = DatasetGenerator(seed=42)
        entries = gen.generate(10)
        assert len(entries) == 10

        pert_engine = PerturbationEngine(seed=42)
        all_perts = []
        for entry in entries:
            perts = pert_engine.generate_all(entry, max_perturbations=2)
            all_perts.extend(perts)

        assert len(all_perts) > 0

        # Run structured workflow
        wf = StructuredSingleWorkflow(llm_provider=None)
        entry_map = {e.requirement_id: e for e in entries}
        y_true, y_pred, y_prob = [], [], []

        for p in all_perts:
            base = entry_map.get(p.base_requirement_id)
            if not base:
                continue
            result = wf.run(base.original_text, p.perturbed_text)
            y_true.append(p.drift_label)
            y_pred.append(1 if result.drift_detected else 0)
            y_prob.append(result.confidence)

        metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)
        assert metrics.n_samples > 0
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1 <= 1
