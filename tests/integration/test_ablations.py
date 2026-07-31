"""Tests for ablation framework."""

from idraak.workflows.ablations import AblationConfig, AblationWorkflow, get_all_ablation_configs


class TestAblationWorkflow:
    def test_no_critic_ablation(self):
        config = AblationConfig(name="test_no_critic", description="test", disable_critic=True)
        wf = AblationWorkflow(ablation_config=config)
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system should respond within 50 ms.",
            requirement_id="REQ-001",
        )
        assert "ablation" in result.workflow

    def test_only_srr_ablation(self):
        config = AblationConfig(name="test_srr", description="test", use_only_srr_comparison=True)
        wf = AblationWorkflow(ablation_config=config)
        result = wf.run(
            original_text="The system shall respond within 5 ms.",
            candidate_text="The system shall respond within 5 ms.",
        )
        assert not result.drift_detected

    def test_get_all_configs(self):
        configs = get_all_ablation_configs()
        assert len(configs) >= 8
        assert "no_critic" in configs
        assert "no_evidence" in configs
        assert "only_srr" in configs

    def test_all_ablations_run(self):
        """Smoke test that all predefined ablations can execute."""
        for name, config in get_all_ablation_configs().items():
            wf = AblationWorkflow(ablation_config=config)
            result = wf.run(
                original_text="The controller shall assert ready within 3 clock cycles.",
                candidate_text="The controller should assert ready within 30 clock cycles.",
                requirement_id=f"REQ-ABL-{name}",
            )
            assert "ablation" in result.workflow or result.workflow in ("direct_judge", "structured_single")
