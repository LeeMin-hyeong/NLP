from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from app.evaluation import build_comparison_table, evaluate_model, load_eval_dataset
from app.model_runner import HFModelRunner, MockModelRunner
from app.pipeline import ConversationToSpecPipeline, ROBUSTNESS_PROFILE_CONFIGS
from app.progress import ConsoleProgressReporter
from app.prompt_builder import load_prompt_config
from app.utils import ensure_dir, load_yaml_file, slugify, write_json_file, write_text_file


MODELS_CONFIG_PATH = Path("configs/models.yaml")
PROMPTS_CONFIG_PATH = Path("configs/prompts.yaml")
PIPELINE_MODES = ("chain", "single_shot")
EXPERIMENT_SUITES = ("rq2", "rq4")


def _resolve_model_alias(input_name: str, models_config: dict[str, Any]) -> tuple[str, str]:
    models = models_config.get("models", {})
    if input_name in models:
        return input_name, str(models[input_name]["hf_repo_id"])

    for alias, cfg in models.items():
        if str(cfg.get("hf_repo_id", "")) == input_name:
            return alias, input_name

    # Direct repository id support.
    return slugify(input_name), input_name


def _build_pipeline(
    *,
    use_mock: bool,
    model_name: str | None,
    prompt_config: dict[str, Any],
    generation_config: dict[str, Any],
    models_config: dict[str, Any],
    pipeline_mode: str = "chain",
    robustness_profile: str | None = None,
) -> tuple[str, ConversationToSpecPipeline]:
    if use_mock:
        runner = MockModelRunner()
        return "mock", ConversationToSpecPipeline(
            runner=runner,
            prompt_config=prompt_config,
            generation_config=generation_config,
            pipeline_mode=pipeline_mode,
            robustness_profile=robustness_profile,
        )

    if not model_name:
        raise ValueError("A model name is required unless --mock is used.")

    alias, hf_repo_id = _resolve_model_alias(model_name, models_config)
    runner = HFModelRunner(hf_repo_id)
    return alias, ConversationToSpecPipeline(
        runner=runner,
        prompt_config=prompt_config,
        generation_config=generation_config,
        pipeline_mode=pipeline_mode,
        robustness_profile=robustness_profile,
    )


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _experiment_run_root(args: argparse.Namespace) -> Path:
    run_id = str(args.run_id or "").strip() or _timestamp_run_id()
    return ensure_dir(Path(args.experiment_root) / run_id)


def _model_repo_id(model_name: str | None, models_config: dict[str, Any]) -> str | None:
    if not model_name or model_name == "mock":
        return None
    _, repo_id = _resolve_model_alias(model_name, models_config)
    return repo_id


def _run_metadata(
    *,
    args: argparse.Namespace,
    model_alias: str,
    model_name: str | None,
    models_config: dict[str, Any],
    generation_config: dict[str, Any],
    dataset_path: Path | None,
    output_dir: Path,
    pipeline_mode: str,
    ablation_profile: str | None,
    experiment_run_root: Path | None,
    suite: str | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    prompt_path = PROMPTS_CONFIG_PATH
    model_path = MODELS_CONFIG_PATH
    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": suite,
        "variant_id": variant_id,
        "pipeline_mode": pipeline_mode,
        "ablation_profile": ablation_profile,
        "model_alias": model_alias,
        "model_input": model_name,
        "hf_repo_id": _model_repo_id(model_name, models_config),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_sha256": _sha256_file(dataset_path) if dataset_path else None,
        "prompt_config_path": str(prompt_path),
        "prompt_config_sha256": _sha256_file(prompt_path),
        "models_config_path": str(model_path),
        "models_config_sha256": _sha256_file(model_path),
        "generation_config": generation_config,
        "robustness_profile_config": ROBUSTNESS_PROFILE_CONFIGS.get(ablation_profile or ""),
        "output_dir": str(output_dir),
        "experiment_run_root": str(experiment_run_root) if experiment_run_root else None,
        "argv": sys.argv,
    }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation-to-Spec CLI")
    parser.add_argument("--input", type=str, help="Path to input transcript (.txt/.md)")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, help="Model alias or Hugging Face repo id")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock model")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation mode")
    parser.add_argument("--dataset", type=str, help="Evaluation dataset path (JSON)")
    parser.add_argument("--all-models", action="store_true", help="Evaluate all configured models")
    parser.add_argument(
        "--pipeline-mode",
        choices=PIPELINE_MODES,
        default="chain",
        help="Pipeline architecture to run.",
    )
    parser.add_argument(
        "--ablation-profile",
        choices=tuple(ROBUSTNESS_PROFILE_CONFIGS.keys()),
        help="Robustness profile for RQ4 ablation runs.",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Write evaluation artifacts under experiments/runs/<timestamp>/.",
    )
    parser.add_argument(
        "--experiment-root",
        type=str,
        default="experiments/runs",
        help="Root directory for timestamped experiment runs.",
    )
    parser.add_argument("--run-id", type=str, help="Optional timestamp/run id to reuse.")
    parser.add_argument(
        "--experiment-suite",
        choices=EXPERIMENT_SUITES,
        help="Run a predefined timestamped experiment suite.",
    )
    return parser.parse_args()


def _run_single(args: argparse.Namespace, models_config: dict[str, Any], prompt_config: dict[str, Any]) -> int:
    if not args.input:
        print("Error: --input is required for single-run mode.")
        return 2
    if args.mock and args.model:
        print("Error: use either --mock or --model, not both.")
        return 2
    if not args.mock and not args.model:
        args.model = str(models_config.get("default_model", ""))

    generation_config = models_config.get("generation", {})
    output_dir = ensure_dir(Path(args.output))
    reporter = ConsoleProgressReporter()

    try:
        _, pipeline = _build_pipeline(
            use_mock=args.mock,
            model_name=args.model,
            prompt_config=prompt_config,
            generation_config=generation_config,
            models_config=models_config,
            pipeline_mode=args.pipeline_mode,
            robustness_profile=args.ablation_profile,
        )
        run = pipeline.run_file(Path(args.input), output_dir, progress_reporter=reporter)
        json_path = run.output_json_path or (output_dir / "spec.json")
        md_path = run.output_md_path or (output_dir / "spec.md")
        print(f"Saved JSON: {json_path}")
        print(f"Saved Markdown: {md_path}")
        print(f"Status: {run.status} (retry_count={run.retry_count})")
        if run.semantic_warnings:
            print(f"Semantic warnings: {len(run.semantic_warnings)}")
        if not run.success:
            error_log = output_dir / "error.log"
            write_text_file(error_log, run.error_message or "Invalid structured output.")
            print(
                "Error: Failed to generate a valid spec output. "
                f"Details recorded at {error_log}: {run.error_message or 'invalid output'}"
            )
            return 1
        return 0
    except Exception as exc:
        error_log = output_dir / "error.log"
        write_text_file(error_log, str(exc))
        print(
            "Error: Failed to generate a valid spec output. "
            f"Details recorded at {error_log}: {exc}"
        )
        return 1


def _suite_variants(suite: str, model_suffix: str) -> list[dict[str, str]]:
    if suite == "rq2":
        return [
            {
                "variant_id": f"RQ2-A1_chain_{model_suffix}",
                "pipeline_mode": "chain",
                "ablation_profile": "FullChain",
            },
            {
                "variant_id": f"RQ2-A2_single_shot_{model_suffix}",
                "pipeline_mode": "single_shot",
                "ablation_profile": "FullChain",
            },
        ]
    if suite == "rq4":
        return [
            {
                "variant_id": f"RQ4-FullChain_{model_suffix}",
                "pipeline_mode": "chain",
                "ablation_profile": "FullChain",
            },
            {
                "variant_id": f"RQ4-NoRetry_{model_suffix}",
                "pipeline_mode": "chain",
                "ablation_profile": "NoRetry",
            },
            {
                "variant_id": f"RQ4-NoSemanticVerify_{model_suffix}",
                "pipeline_mode": "chain",
                "ablation_profile": "NoSemanticVerify",
            },
            {
                "variant_id": f"RQ4-StrictRaw_{model_suffix}",
                "pipeline_mode": "chain",
                "ablation_profile": "StrictRaw",
            },
        ]
    raise ValueError(f"Unsupported experiment suite: {suite}")


def _run_experiment_suite(
    args: argparse.Namespace,
    models_config: dict[str, Any],
    prompt_config: dict[str, Any],
) -> int:
    if not args.dataset:
        print("Error: --dataset is required for --experiment-suite.")
        return 2
    if args.mock:
        print("Error: --experiment-suite must use a real local HF model, not --mock.")
        return 2
    if args.all_models:
        print("Error: --experiment-suite cannot be combined with --all-models.")
        return 2

    dataset_path = Path(args.dataset)
    dataset = load_eval_dataset(dataset_path)
    generation_config = models_config.get("generation", {})
    model_name = args.model or str(models_config.get("default_model", "gemma_3_1b_it"))
    model_suffix, _ = _resolve_model_alias(model_name, models_config)
    suite_variants = _suite_variants(str(args.experiment_suite), model_suffix)
    run_root = _experiment_run_root(args)
    suite_dir = ensure_dir(run_root / str(args.experiment_suite))
    eval_root = ensure_dir(suite_dir / "eval_output")
    reporter = ConsoleProgressReporter()
    all_reports: dict[str, dict[str, Any]] = {}

    suite_metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "suite": args.experiment_suite,
        "model": model_name,
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "variants": suite_variants,
        "run_root": str(run_root),
    }
    write_json_file(suite_dir / "suite_config.json", suite_metadata)

    for index, variant in enumerate(suite_variants, start=1):
        variant_id = variant["variant_id"]
        output_dir = ensure_dir(eval_root / slugify(variant_id))
        try:
            reporter.message(
                f"Experiment {index}/{len(suite_variants)} "
                f"[{variant_id}] started"
            )
            resolved_alias, pipeline = _build_pipeline(
                use_mock=False,
                model_name=model_name,
                prompt_config=prompt_config,
                generation_config=generation_config,
                models_config=models_config,
                pipeline_mode=variant["pipeline_mode"],
                robustness_profile=variant["ablation_profile"],
            )
            metadata = _run_metadata(
                args=args,
                model_alias=resolved_alias,
                model_name=model_name,
                models_config=models_config,
                generation_config=generation_config,
                dataset_path=dataset_path,
                output_dir=output_dir,
                pipeline_mode=variant["pipeline_mode"],
                ablation_profile=variant["ablation_profile"],
                experiment_run_root=run_root,
                suite=str(args.experiment_suite),
                variant_id=variant_id,
            )
            report = evaluate_model(
                model_label=f"{resolved_alias}:{variant_id}",
                pipeline=pipeline,
                samples=dataset,
                output_dir=output_dir,
                progress_reporter=reporter,
                run_metadata=metadata,
            )
            reporter.message(f"Experiment {index}/{len(suite_variants)} [{variant_id}] finished")
        except Exception as exc:
            report = {
                "model": f"{model_name}:{variant_id}",
                "metrics": {
                    "sample_count": len(dataset),
                    "functional_f1": 0.0,
                    "non_functional_f1": 0.0,
                    "requirement_type_macro_f1": 0.0,
                    "open_question_recall": 0.0,
                    "follow_up_question_coverage": 0.0,
                    "hallucination_rate": 0.0,
                    "schema_validity_rate": 0.0,
                    "avg_latency_sec": 0.0,
                },
                "error": str(exc),
            }
            write_text_file(output_dir / "error.log", str(exc))
        all_reports[variant_id] = report

    comparison_json = suite_dir / "comparison_results.json"
    comparison_md = suite_dir / "comparison_table.md"
    write_json_file(comparison_json, all_reports)
    write_text_file(comparison_md, build_comparison_table(all_reports))
    print(f"Saved experiment run: {run_root}")
    print(f"Saved comparison JSON: {comparison_json}")
    print(f"Saved comparison table: {comparison_md}")
    return 0


def _run_evaluate(
    args: argparse.Namespace, models_config: dict[str, Any], prompt_config: dict[str, Any]
) -> int:
    if not args.dataset:
        print("Error: --dataset is required when --evaluate is set.")
        return 2
    if args.mock and args.all_models:
        print("Error: --mock cannot be combined with --all-models.")
        return 2
    if not args.mock and not args.all_models and not args.model:
        args.model = str(models_config.get("default_model", ""))

    dataset = load_eval_dataset(Path(args.dataset))
    generation_config = models_config.get("generation", {})
    experiment_run_root = _experiment_run_root(args) if args.experiment else None
    eval_root = ensure_dir(
        (experiment_run_root / "eval_output") if experiment_run_root else Path("eval_output")
    )
    reporter = ConsoleProgressReporter()

    if args.all_models:
        model_aliases = list(models_config.get("compare_models", []))
        if not model_aliases:
            model_aliases = list(models_config.get("models", {}).keys())
        if not model_aliases:
            print("Error: no models configured in configs/models.yaml.")
            return 2

        all_reports: dict[str, dict[str, Any]] = {}
        for model_index, alias in enumerate(model_aliases, start=1):
            alias_slug_parts = [alias, args.pipeline_mode]
            if args.ablation_profile:
                alias_slug_parts.append(args.ablation_profile)
            model_output_dir = ensure_dir(eval_root / slugify("__".join(alias_slug_parts)))
            try:
                reporter.message(f"Model {model_index}/{len(model_aliases)} [{alias}] started")
                resolved_alias, pipeline = _build_pipeline(
                    use_mock=False,
                    model_name=alias,
                    prompt_config=prompt_config,
                    generation_config=generation_config,
                    models_config=models_config,
                    pipeline_mode=args.pipeline_mode,
                    robustness_profile=args.ablation_profile,
                )
                metadata = _run_metadata(
                    args=args,
                    model_alias=resolved_alias,
                    model_name=alias,
                    models_config=models_config,
                    generation_config=generation_config,
                    dataset_path=Path(args.dataset),
                    output_dir=model_output_dir,
                    pipeline_mode=args.pipeline_mode,
                    ablation_profile=args.ablation_profile,
                    experiment_run_root=experiment_run_root,
                )
                report = evaluate_model(
                    model_label=resolved_alias,
                    pipeline=pipeline,
                    samples=dataset,
                    output_dir=model_output_dir,
                    progress_reporter=reporter,
                    run_metadata=metadata,
                )
                reporter.message(f"Model {model_index}/{len(model_aliases)} [{alias}] finished")
            except Exception as exc:
                report = {
                    "model": alias,
                    "metrics": {
                        "sample_count": len(dataset),
                        "functional_f1": 0.0,
                        "non_functional_f1": 0.0,
                        "requirement_type_macro_f1": 0.0,
                        "open_question_recall": 0.0,
                        "follow_up_question_coverage": 0.0,
                        "hallucination_rate": 0.0,
                        "schema_validity_rate": 0.0,
                        "avg_latency_sec": 0.0,
                    },
                    "error": str(exc),
                }
                write_text_file(model_output_dir / "error.log", str(exc))
            all_reports[alias] = report

        comparison_json = eval_root / "comparison_results.json"
        comparison_md = eval_root / "comparison_table.md"
        write_json_file(comparison_json, all_reports)
        write_text_file(comparison_md, build_comparison_table(all_reports))
        print(f"Saved comparison JSON: {comparison_json}")
        print(f"Saved comparison table: {comparison_md}")
        return 0

    use_mock = bool(args.mock)
    model_name = "mock" if use_mock else args.model
    output_slug_parts = [str(model_name), args.pipeline_mode]
    if args.ablation_profile:
        output_slug_parts.append(args.ablation_profile)
    output_dir = ensure_dir(eval_root / slugify("__".join(output_slug_parts)))
    try:
        resolved_alias, pipeline = _build_pipeline(
            use_mock=use_mock,
            model_name=model_name,
            prompt_config=prompt_config,
            generation_config=generation_config,
            models_config=models_config,
            pipeline_mode=args.pipeline_mode,
            robustness_profile=args.ablation_profile,
        )
        metadata = _run_metadata(
            args=args,
            model_alias=resolved_alias,
            model_name=model_name,
            models_config=models_config,
            generation_config=generation_config,
            dataset_path=Path(args.dataset),
            output_dir=output_dir,
            pipeline_mode=args.pipeline_mode,
            ablation_profile=args.ablation_profile,
            experiment_run_root=experiment_run_root,
        )
        report = evaluate_model(
            model_label=resolved_alias,
            pipeline=pipeline,
            samples=dataset,
            output_dir=output_dir,
            progress_reporter=reporter,
            run_metadata=metadata,
        )
        print(f"Saved metrics: {output_dir / 'metrics.json'}")
        if experiment_run_root:
            print(f"Saved experiment run: {experiment_run_root}")
        print(f"Model: {report.get('model')}")
        return 0
    except Exception as exc:
        write_text_file(output_dir / "error.log", str(exc))
        print(f"Error: evaluation failed for {model_name}: {exc}")
        return 1


def main() -> int:
    args = parse_args()
    models_config = load_yaml_file(MODELS_CONFIG_PATH)
    prompt_config = load_prompt_config(PROMPTS_CONFIG_PATH)

    if args.experiment_suite:
        return _run_experiment_suite(args, models_config, prompt_config)
    if args.evaluate:
        return _run_evaluate(args, models_config, prompt_config)
    return _run_single(args, models_config, prompt_config)


if __name__ == "__main__":
    sys.exit(main())
