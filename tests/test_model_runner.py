from app.model_runner import HuggingFaceModelRunner, MockModelRunner, create_model_runner


def test_create_model_runner_returns_mock_for_mock_type() -> None:
    cfg = {
        "default_model": "mock",
        "models": {
            "mock": {
                "type": "mock",
            }
        },
    }
    runner = create_model_runner(model_key="mock", models_config=cfg, use_mock=False)
    assert isinstance(runner, MockModelRunner)


def test_create_model_runner_returns_hf_runner() -> None:
    cfg = {
        "default_model": "qwen",
        "generation_defaults": {"max_new_tokens": 256},
        "models": {
            "qwen": {
                "type": "huggingface",
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "generation": {"max_new_tokens": 400},
            }
        },
    }

    runner = create_model_runner(model_key="qwen", models_config=cfg, use_mock=False)

    assert isinstance(runner, HuggingFaceModelRunner)
    assert runner.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert runner.generation_config["max_new_tokens"] == 400
