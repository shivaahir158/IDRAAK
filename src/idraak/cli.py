"""IDRAAK command-line interface."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="idraak", help="IDRAAK: Semantic Drift Detection Framework")
console = Console()


@app.command()
def generate_dataset(
    n: int = typer.Option(300, help="Number of requirements to generate"),
    output: str = typer.Option("data/raw/requirements.jsonl", help="Output path"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Generate synthetic technical requirement dataset."""
    from idraak.datasets.generator import DatasetGenerator
    from idraak.perturbations.engine import PerturbationEngine
    from idraak.utils.seed import set_global_seed

    set_global_seed(seed)
    gen = DatasetGenerator(seed=seed)
    entries = gen.generate(n)
    gen.save(entries, output)
    console.print(f"[green]Generated {len(entries)} requirements -> {output}[/green]")

    # Generate perturbations
    pert_engine = PerturbationEngine(seed=seed)
    all_perts = []
    for entry in entries:
        perts = pert_engine.generate_all(entry, max_perturbations=3)
        all_perts.extend(perts)

    pert_path = Path(output).parent / "perturbations.jsonl"
    with open(pert_path, "w") as f:
        for p in all_perts:
            f.write(p.model_dump_json() + "\n")
    console.print(f"[green]Generated {len(all_perts)} perturbations -> {pert_path}[/green]")

    # Stats
    drift_count = sum(1 for p in all_perts if p.drift_label == 1)
    no_drift_count = sum(1 for p in all_perts if p.drift_label == 0)
    console.print(f"  Drift examples: {drift_count}")
    console.print(f"  No-drift examples: {no_drift_count}")


def _build_providers(provider: str) -> tuple:
    """Build LLM and translation providers based on provider name.

    Returns (llm_provider, translation_provider).
    """
    if provider == "openai":
        from idraak.providers.openai_provider import OpenAILLMProvider, OpenAITranslationProvider
        return OpenAILLMProvider(), OpenAITranslationProvider()
    elif provider == "mock":
        from idraak.providers.mock import MockLLMProvider, MockTranslationProvider
        return MockLLMProvider(), MockTranslationProvider()
    else:
        return None, None


def _build_workflow(workflow: str, llm_provider, translation_provider):
    """Instantiate a workflow by name."""
    if workflow == "direct_judge":
        from idraak.workflows.direct_judge import DirectJudgeWorkflow
        return DirectJudgeWorkflow(llm_provider=llm_provider)
    elif workflow == "structured_single":
        from idraak.workflows.structured_single import StructuredSingleWorkflow
        return StructuredSingleWorkflow(llm_provider=llm_provider)
    elif workflow == "full_idraak":
        from idraak.workflows.full_idraak import FullIDRAAKWorkflow
        return FullIDRAAKWorkflow(
            translation_provider=translation_provider,
            llm_provider=llm_provider,
        )
    else:
        raise ValueError(f"Unknown workflow: {workflow}")


@app.command()
def evaluate(
    dataset: str = typer.Option("data/raw/requirements.jsonl", help="Dataset path"),
    perturbations: str = typer.Option("data/raw/perturbations.jsonl", help="Perturbations path"),
    workflow: str = typer.Option("structured_single", help="Workflow: direct_judge, structured_single, full_idraak"),
    provider: str = typer.Option("none", help="LLM provider: none, mock, openai"),
    model: str = typer.Option("gpt-4o-mini", help="Model name (for openai provider)"),
    output_dir: str = typer.Option("reports", help="Output directory"),
    max_samples: int = typer.Option(0, help="Max perturbations to evaluate (0=all)"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Evaluate drift detection on the dataset."""
    import numpy as np

    from idraak.datasets.generator import DatasetGenerator
    from idraak.evaluation.metrics import ClassificationMetrics
    from idraak.reporting.plots import PlotGenerator
    from idraak.reporting.tables import TableGenerator
    from idraak.schemas.dataset import DatasetEntry, PerturbationRecord
    from idraak.tracking.tracker import ExperimentTracker
    from idraak.utils.seed import set_global_seed

    set_global_seed(seed)

    # Build providers
    llm_provider, translation_provider = _build_providers(provider)
    if provider == "openai":
        from idraak.providers.openai_provider import OpenAILLMProvider
        llm_provider = OpenAILLMProvider(model=model)

    # Build workflow
    wf = _build_workflow(workflow, llm_provider, translation_provider)
    wf_label = f"{workflow}" + (f"/{provider}/{model}" if provider != "none" else "/deterministic")

    # Load data
    entries = DatasetGenerator.load(dataset)
    perts: list[PerturbationRecord] = []
    with open(perturbations) as f:
        for line in f:
            if line.strip():
                perts.append(PerturbationRecord.model_validate_json(line))

    if max_samples > 0:
        perts = perts[:max_samples]

    console.print(f"[bold]{wf_label}[/bold]")
    console.print(f"Loaded {len(entries)} requirements, {len(perts)} perturbations")

    # Build entry lookup
    entry_map = {e.requirement_id: e for e in entries}

    # Run workflow
    y_true, y_pred, y_prob = [], [], []
    results = []
    start_time = time.time()
    errors = 0

    for i, pert in enumerate(perts):
        base_entry = entry_map.get(pert.base_requirement_id)
        if not base_entry:
            continue

        try:
            result = wf.run(
                original_text=base_entry.original_text,
                candidate_text=pert.perturbed_text,
                requirement_id=pert.requirement_id,
            )
        except Exception as e:
            errors += 1
            console.print(f"[red]Error on {pert.requirement_id}: {e}[/red]")
            continue

        y_true.append(pert.drift_label)
        y_pred.append(1 if result.drift_detected else 0)
        y_prob.append(result.confidence)
        results.append({
            "requirement_id": pert.requirement_id,
            "base_id": pert.base_requirement_id,
            "drift_type": pert.drift_type,
            "gold_label": pert.drift_label,
            "pred_label": 1 if result.drift_detected else 0,
            "confidence": result.confidence,
            "n_diffs": len(result.field_differences),
            "workflow": workflow,
            "provider": provider,
            "model": model if provider == "openai" else "n/a",
        })

        if (i + 1) % 50 == 0:
            console.print(f"  Processed {i+1}/{len(perts)}...")

    total_time = time.time() - start_time
    if errors:
        console.print(f"[yellow]Completed with {errors} errors[/yellow]")

    # Compute metrics
    metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)
    console.print(f"\n[bold]Results ({wf_label}):[/bold]")

    table = Table(title="Classification Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in metrics.to_dict().items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    console.print(table)
    console.print(f"Total evaluation time: {total_time:.1f}s")

    # Save results
    out_subdir = Path(output_dir) / f"{workflow}_{provider}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    results_path = out_subdir / "evaluation_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Generate tables
    tg = TableGenerator(str(out_subdir))
    tg.metrics_table({wf_label: metrics.to_dict()})

    # Generate plots
    pg = PlotGenerator(str(out_subdir))
    if metrics.confusion_matrix:
        pg.confusion_matrix_plot(metrics.confusion_matrix)

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    if len(y_prob_arr) > 0:
        pg.reliability_diagram(y_true_arr, y_prob_arr)
        correct_mask = np.array(y_true) == np.array(y_pred)
        if correct_mask.any() and (~correct_mask).any():
            pg.confidence_distribution(y_prob_arr[correct_mask], y_prob_arr[~correct_mask])

    # Per drift type metrics
    drift_types = [p.drift_type or "paraphrase" for p in perts if entry_map.get(p.base_requirement_id)]
    if drift_types and len(drift_types) == len(y_true):
        per_type = ClassificationMetrics.per_group_metrics(y_true, y_pred, drift_types, y_prob)
        type_f1 = {k: v.f1 for k, v in per_type.items()}
        pg.f1_by_drift_type(type_f1)
        tg.per_language_table(
            {k: v.to_dict() for k, v in per_type.items()},
            name="per_drift_type",
        )

    # Track
    tracker = ExperimentTracker()
    tracker.log_config({
        "workflow": workflow, "provider": provider, "model": model,
        "seed": seed, "n_perturbations": len(perts), "errors": errors,
    })
    tracker.log_metrics(metrics.to_dict())
    tracker.save_summary(metrics.to_dict())

    console.print(f"\n[green]Results saved to {out_subdir}/[/green]")


@app.command()
def run(
    config: str = typer.Option("configs/default.yaml", help="Config file"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run complete end-to-end pipeline: generate -> evaluate -> report."""
    from idraak.utils.seed import set_global_seed

    set_global_seed(seed)
    console.print("[bold]IDRAAK End-to-End Pipeline[/bold]")

    # Step 1: Generate dataset
    console.print("\n[bold blue]Step 1: Generating dataset...[/bold blue]")
    generate_dataset(n=50, output="data/raw/requirements.jsonl", seed=seed)

    # Step 2: Evaluate
    console.print("\n[bold blue]Step 2: Running evaluation...[/bold blue]")
    evaluate(
        dataset="data/raw/requirements.jsonl",
        perturbations="data/raw/perturbations.jsonl",
        workflow="structured_single",
        provider="none",
        model="gpt-4o-mini",
        output_dir="reports",
        max_samples=0,
        seed=seed,
    )

    console.print("\n[bold green]Pipeline complete![/bold green]")


@app.command()
def experiment_matrix(
    dataset: str = typer.Option("data/raw/requirements_300.jsonl", help="Dataset path"),
    perturbations: str = typer.Option("data/raw/perturbations.jsonl", help="Perturbations path"),
    model: str = typer.Option("gpt-4o-mini", help="OpenAI model to use"),
    max_samples: int = typer.Option(0, help="Max perturbations per experiment (0=all)"),
    output_dir: str = typer.Option("reports/experiments", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run full experiment matrix: all workflows x deterministic + OpenAI."""
    import numpy as np

    from idraak.datasets.generator import DatasetGenerator
    from idraak.evaluation.metrics import ClassificationMetrics
    from idraak.reporting.plots import PlotGenerator
    from idraak.reporting.tables import TableGenerator
    from idraak.schemas.dataset import PerturbationRecord
    from idraak.utils.seed import set_global_seed

    set_global_seed(seed)

    experiments = [
        ("structured_single", "none"),
        ("structured_single", "openai"),
        ("direct_judge", "openai"),
        ("full_idraak", "none"),
        ("full_idraak", "openai"),
    ]

    # Load data once
    entries = DatasetGenerator.load(dataset)
    perts: list[PerturbationRecord] = []
    with open(perturbations) as f:
        for line in f:
            if line.strip():
                perts.append(PerturbationRecord.model_validate_json(line))

    if max_samples > 0:
        perts = perts[:max_samples]

    entry_map = {e.requirement_id: e for e in entries}
    console.print(f"[bold]Experiment Matrix[/bold] — {len(perts)} perturbations, model={model}")

    all_metrics: dict[str, dict] = {}

    for workflow, provider in experiments:
        label = f"{workflow}/{provider}" + (f"/{model}" if provider == "openai" else "")
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]Running: {label}[/bold blue]")

        llm_provider, translation_provider = _build_providers(provider)
        if provider == "openai":
            from idraak.providers.openai_provider import OpenAILLMProvider
            llm_provider = OpenAILLMProvider(model=model)

        wf = _build_workflow(workflow, llm_provider, translation_provider)

        y_true, y_pred, y_prob = [], [], []
        errors = 0
        start_time = time.time()

        for i, pert in enumerate(perts):
            base_entry = entry_map.get(pert.base_requirement_id)
            if not base_entry:
                continue
            try:
                result = wf.run(
                    original_text=base_entry.original_text,
                    candidate_text=pert.perturbed_text,
                    requirement_id=pert.requirement_id,
                )
                y_true.append(pert.drift_label)
                y_pred.append(1 if result.drift_detected else 0)
                y_prob.append(result.confidence)
            except Exception as e:
                errors += 1
                if errors <= 3:
                    console.print(f"  [red]Error: {e}[/red]")

            if (i + 1) % 100 == 0:
                console.print(f"  Processed {i+1}/{len(perts)}...")

        elapsed = time.time() - start_time
        metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)
        all_metrics[label] = metrics.to_dict()
        all_metrics[label]["elapsed_seconds"] = round(elapsed, 1)
        all_metrics[label]["errors"] = errors

        console.print(
            f"  [green]F1={metrics.f1:.4f}  Acc={metrics.accuracy:.4f}  "
            f"MCC={metrics.mcc:.4f}  ({elapsed:.1f}s, {errors} errors)[/green]"
        )

    # Save comparison
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tg = TableGenerator(output_dir)
    tg.metrics_table(all_metrics, name="experiment_matrix")

    pg = PlotGenerator(output_dir)
    pg.method_comparison(all_metrics, name="experiment_matrix")

    # Save raw metrics as JSON
    results_path = Path(output_dir) / "experiment_matrix.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    console.print(f"\n[bold green]Experiment matrix complete! Results in {output_dir}/[/bold green]")

    # Print summary table
    table = Table(title="Experiment Matrix Results")
    table.add_column("Experiment")
    table.add_column("F1", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("MCC", justify="right")
    table.add_column("Time", justify="right")
    for label, m in all_metrics.items():
        table.add_row(
            label,
            f"{m.get('f1', 0):.4f}",
            f"{m.get('accuracy', 0):.4f}",
            f"{m.get('mcc', 0):.4f}",
            f"{m.get('elapsed_seconds', 0):.1f}s",
        )
    console.print(table)


@app.command()
def demo() -> None:
    """Run a quick demonstration with mock data."""
    from idraak.datasets.generator import DatasetGenerator
    from idraak.drift.comparator import SRRComparator
    from idraak.extraction.deterministic import DeterministicExtractor
    from idraak.perturbations.engine import PerturbationEngine
    from idraak.providers.mock import MockTranslationProvider
    from idraak.workflows.full_idraak import FullIDRAAKWorkflow

    console.print("[bold]IDRAAK Quick Demo[/bold]\n")

    # Generate a sample requirement
    gen = DatasetGenerator(seed=42)
    entries = gen.generate(5)

    for entry in entries[:3]:
        console.print(f"[cyan]Requirement:[/cyan] {entry.requirement_id}")
        console.print(f"  Text: {entry.original_text}")
        console.print(f"  Domain: {entry.domain} | Category: {entry.category}")

        # Extract SRR
        extractor = DeterministicExtractor()
        srr = extractor.extract(entry.original_text, "en", entry.requirement_id)
        console.print(f"  Modality: {srr.modality}")
        console.print(f"  Polarity: {srr.polarity}")
        console.print(f"  Numerical: {len(srr.numerical_constraints)} constraints")
        console.print(f"  Temporal: {len(srr.temporal_constraints)} constraints")

        # Generate perturbation
        pert_engine = PerturbationEngine(seed=42)
        perts = pert_engine.generate_all(entry, max_perturbations=2)

        for p in perts[:1]:
            console.print(f"\n  [yellow]Perturbation ({p.drift_type or 'paraphrase'}):[/yellow]")
            console.print(f"    {p.perturbed_text}")

            # Run full IDRAAK workflow
            mock_provider = MockTranslationProvider()
            wf = FullIDRAAKWorkflow(translation_provider=mock_provider)
            result = wf.run(
                original_text=entry.original_text,
                candidate_text=p.perturbed_text,
                requirement_id=entry.requirement_id,
            )
            console.print(f"    Drift detected: {result.drift_detected}")
            console.print(f"    Severity: {result.severity.value}")
            console.print(f"    Confidence: {result.confidence:.2f}")
            console.print(f"    Gold label: {'drift' if p.drift_label == 1 else 'no drift'}")

        console.print()


@app.command()
def run_ablation(
    dataset: str = typer.Option("data/raw/requirements.jsonl", help="Dataset path"),
    perturbations: str = typer.Option("data/raw/perturbations.jsonl", help="Perturbations path"),
    ablation: str = typer.Option("all", help="Ablation name or 'all'"),
    output_dir: str = typer.Option("reports/ablations", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run ablation studies."""
    import numpy as np

    from idraak.datasets.generator import DatasetGenerator
    from idraak.evaluation.metrics import ClassificationMetrics
    from idraak.reporting.plots import PlotGenerator
    from idraak.reporting.tables import TableGenerator
    from idraak.schemas.dataset import PerturbationRecord
    from idraak.utils.seed import set_global_seed
    from idraak.workflows.ablations import AblationWorkflow, get_all_ablation_configs

    set_global_seed(seed)

    # Load data
    entries = DatasetGenerator.load(dataset)
    perts: list[PerturbationRecord] = []
    with open(perturbations) as f:
        for line in f:
            if line.strip():
                perts.append(PerturbationRecord.model_validate_json(line))

    entry_map = {e.requirement_id: e for e in entries}

    # Get ablation configs
    all_configs = get_all_ablation_configs()
    if ablation == "all":
        configs_to_run = all_configs
    else:
        if ablation not in all_configs:
            console.print(f"[red]Unknown ablation: {ablation}[/red]")
            console.print(f"Available: {', '.join(all_configs.keys())}")
            raise typer.Exit(1)
        configs_to_run = {ablation: all_configs[ablation]}

    all_metrics: dict[str, dict[str, float]] = {}

    for abl_name, abl_config in configs_to_run.items():
        console.print(f"\n[bold blue]Running ablation: {abl_name}[/bold blue] — {abl_config.description}")
        wf = AblationWorkflow(ablation_config=abl_config)

        y_true, y_pred, y_prob = [], [], []
        for pert in perts:
            base = entry_map.get(pert.base_requirement_id)
            if not base:
                continue
            result = wf.run(base.original_text, pert.perturbed_text, pert.requirement_id)
            y_true.append(pert.drift_label)
            y_pred.append(1 if result.drift_detected else 0)
            y_prob.append(result.confidence)

        metrics = ClassificationMetrics.compute(y_true, y_pred, y_prob)
        all_metrics[abl_name] = metrics.to_dict()
        console.print(f"  F1: {metrics.f1:.4f} | Accuracy: {metrics.accuracy:.4f} | MCC: {metrics.mcc:.4f}")

    # Save comparison
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tg = TableGenerator(output_dir)
    tg.metrics_table(all_metrics, name="ablation_results")

    pg = PlotGenerator(output_dir)
    pg.method_comparison(all_metrics, name="ablation_comparison")

    console.print(f"\n[green]Ablation results saved to {output_dir}/[/green]")


@app.command()
def error_analysis(
    results_file: str = typer.Option("reports/evaluation_results.jsonl", help="Results file"),
    dataset: str = typer.Option("data/raw/requirements.jsonl", help="Dataset path"),
    output_dir: str = typer.Option("reports", help="Output directory"),
) -> None:
    """Generate error analysis report."""
    from idraak.datasets.generator import DatasetGenerator
    from idraak.evaluation.error_analysis import ErrorAnalyzer

    entries = DatasetGenerator.load(dataset)
    entry_map = {e.requirement_id: e.model_dump() for e in entries}

    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    analyzer = ErrorAnalyzer(output_dir)
    report = analyzer.analyze(results, entry_map)
    analyzer.save_report(report)

    console.print(f"[bold]Error Analysis:[/bold]")
    console.print(f"  Total predictions: {report.total_predictions}")
    console.print(f"  Total errors: {report.total_errors}")
    console.print(f"  False positives: {len(report.false_positives)}")
    console.print(f"  False negatives: {len(report.false_negatives)}")
    console.print(f"\n[green]Report saved to {output_dir}/[/green]")


@app.command()
def generate_report(
    output_dir: str = typer.Option("reports", help="Output directory"),
) -> None:
    """Generate publication-ready tables and plots from existing results."""
    import numpy as np

    from idraak.reporting.plots import PlotGenerator
    from idraak.reporting.tables import TableGenerator

    results_path = Path(output_dir) / "evaluation_results.jsonl"
    if not results_path.exists():
        console.print("[red]No evaluation results found. Run 'evaluate' first.[/red]")
        raise typer.Exit(1)

    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Dataset stats
    dataset_path = Path("data/raw/requirements.jsonl")
    if dataset_path.exists():
        from idraak.datasets.generator import DatasetGenerator
        entries = DatasetGenerator.load(dataset_path)
        tg = TableGenerator(output_dir)
        tg.dataset_stats_table([e.model_dump() for e in entries])
        console.print("[green]Generated dataset statistics table[/green]")

    console.print(f"[green]Reports available in {output_dir}/[/green]")


if __name__ == "__main__":
    app()
