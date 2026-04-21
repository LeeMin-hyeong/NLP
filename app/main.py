from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from app.evaluation import evaluate_model_on_dataset, render_comparison_table
from app.model_runner import available_model_keys, create_model_runner
from app.pipeline import run_pipeline
from app.utils import ensure_directory, load_yaml


def _build_prompt_config(prompts_config: dict, models_config: dict, model_key: Optional[str]) -> dict:
    prompt_payload = dict(prompts_config.get("prompt", prompts_config))

    generation = dict(models_config.get("generation_defaults", {}))
    model_entry = None
    if model_key and model_key in models_config.get("models", {}):
        model_entry = models_config["models"][model_key]
        generation.update(model_entry.get("generation", {}))

    prompt_payload["generation"] = generation

    if model_entry is not None:
        prompt_overrides = model_entry.get("prompt_overrides", {})
        if isinstance(prompt_overrides, dict):
            prompt_payload.update(prompt_overrides)

    context_hint = int(models_config.get("default_context_window_hint", 4096))
    if model_entry is not None:
        context_hint = int(model_entry.get("context_window_hint", context_hint))

    configured_input_cap = prompt_payload.get("max_input_tokens")
    if isinstance(configured_input_cap, int) and configured_input_cap > 0:
        prompt_payload["max_input_tokens"] = min(configured_input_cap, context_hint)
    else:
        prompt_payload["max_input_tokens"] = context_hint

    return prompt_payload


def _resolve_model_keys(args: argparse.Namespace, models_config: dict) -> list[str]:
    if args.mock:
        return ["mock"]

    if args.model:
        return [args.model]

    configured = models_config.get("comparison_models")
    if configured:
        return list(configured)

    keys = available_model_keys(models_config)
    if keys:
        return keys

    default_model = models_config.get("default_model")
    return [default_model] if default_model else ["mock"]


def _run_single_inference(args: argparse.Namespace, models_config: dict, prompts_config: dict) -> int:
    model_key = "mock" if args.mock else args.model or models_config.get("default_model")
    if model_key is None:
        raise ValueError("No model selected. Set --model, --mock, or default_model in configs/models.yaml")

    prompt_config = _build_prompt_config(
        prompts_config=prompts_config,
        models_config=models_config,
        model_key=model_key,
    )

    runner = create_model_runner(
        model_key=model_key,
        models_config=models_config,
        use_mock=args.mock,
    )

    result = run_pipeline(
        input_path=args.input,
        output_dir=args.output,
        runner=runner,
        prompt_config=prompt_config,
    )

    print(f"Model: {result.generation.model_name}")
    print(f"Latency: {result.generation.latency_sec:.3f}s")
    print(f"JSON output: {result.json_path}")
    print(f"Markdown output: {result.markdown_path}")
    return 0


def _run_evaluation(args: argparse.Namespace, models_config: dict, prompts_config: dict) -> int:
    model_keys = _resolve_model_keys(args, models_config=models_config)
    eval_dir = ensure_directory(args.eval_output)

    all_results: list[dict] = []

    for model_key in model_keys:
        prompt_config = _build_prompt_config(
            prompts_config=prompts_config,
            models_config=models_config,
            model_key=model_key,
        )

        try:
            runner = create_model_runner(
                model_key=model_key,
                models_config=models_config,
                use_mock=(model_key == "mock"),
            )
            result = evaluate_model_on_dataset(
                dataset_path=args.dataset,
                runner=runner,
                prompt_config=prompt_config,
                eval_output_dir=eval_dir,
                model_label=model_key,
            )
            all_results.append(result)
        except Exception as exc:  # noqa: BLE001
            all_results.append(
                {
                    "model": model_key,
                    "aggregate_metrics": {},
                    "per_sample": [],
                    "error": str(exc),
                }
            )

    comparison_table = render_comparison_table(all_results)
    print(comparison_table)

    comparison_path = Path(eval_dir) / "comparison_results.json"
    comparison_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    table_path = Path(eval_dir) / "comparison_table.md"
    table_path.write_text(comparison_table + "\n", encoding="utf-8")

    print(f"Saved comparison JSON: {comparison_path}")
    print(f"Saved comparison table: {table_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conversation-to-Spec CLI prototype")
    parser.add_argument("--input", type=Path, help="Path to input conversation (.txt or .md)")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")

    parser.add_argument("--model", type=str, default=None, help="Model key (or HF model name)")
    parser.add_argument("--mock", action="store_true", help="Use mock model runner")

    parser.add_argument("--evaluate", action="store_true", help="Run quantitative evaluation")
    parser.add_argument("--dataset", type=Path, default=Path("dataset/eval_samples.json"), help="Evaluation dataset path")
    parser.add_argument("--eval-output", type=Path, default=Path("eval_output"), help="Evaluation output directory")

    parser.add_argument(
        "--models-config",
        type=Path,
        default=Path("configs/models.yaml"),
        help="Model configuration YAML path",
    )
    parser.add_argument(
        "--prompts-config",
        type=Path,
        default=Path("configs/prompts.yaml"),
        help="Prompt configuration YAML path",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        models_config = load_yaml(args.models_config)
        prompts_config = load_yaml(args.prompts_config)

        if args.evaluate:
            return _run_evaluation(args=args, models_config=models_config, prompts_config=prompts_config)

        if args.input is None:
            parser.error("--input is required unless --evaluate is used")

        return _run_single_inference(args=args, models_config=models_config, prompts_config=prompts_config)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
