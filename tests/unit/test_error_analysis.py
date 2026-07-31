"""Tests for error analysis."""

from idraak.evaluation.error_analysis import ErrorAnalyzer


class TestErrorAnalyzer:
    def test_analyze_with_errors(self):
        results = [
            {"requirement_id": "R1", "gold_label": 1, "pred_label": 0, "confidence": 0.3, "drift_type": "numerical_drift"},
            {"requirement_id": "R2", "gold_label": 0, "pred_label": 1, "confidence": 0.8, "drift_type": None},
            {"requirement_id": "R3", "gold_label": 1, "pred_label": 1, "confidence": 0.9, "drift_type": "modality_drift"},
            {"requirement_id": "R4", "gold_label": 0, "pred_label": 0, "confidence": 0.2, "drift_type": None},
        ]
        analyzer = ErrorAnalyzer("/tmp/test_error_analysis")
        report = analyzer.analyze(results)
        assert report.total_predictions == 4
        assert report.total_errors == 2
        assert len(report.false_positives) == 1
        assert len(report.false_negatives) == 1

    def test_analyze_no_errors(self):
        results = [
            {"requirement_id": "R1", "gold_label": 1, "pred_label": 1},
            {"requirement_id": "R2", "gold_label": 0, "pred_label": 0},
        ]
        analyzer = ErrorAnalyzer("/tmp/test_error_analysis")
        report = analyzer.analyze(results)
        assert report.total_errors == 0

    def test_confidence_binning(self):
        results = [
            {"requirement_id": f"R{i}", "gold_label": 1, "pred_label": 0, "confidence": i * 0.1}
            for i in range(10)
        ]
        analyzer = ErrorAnalyzer("/tmp/test_error_analysis")
        report = analyzer.analyze(results)
        assert sum(report.by_confidence_bin.values()) == 10
