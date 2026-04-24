from app.model_runner import HFModelRunner


class _DummyTokenizer:
    pass


class _RetryingAutoTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def from_pretrained(self, model_name: str, **kwargs):
        self.calls.append((model_name, kwargs))
        if not kwargs:
            raise AttributeError("'list' object has no attribute 'keys'")
        return _DummyTokenizer()


def test_hf_model_runner_retries_tokenizer_load_with_empty_extra_special_tokens():
    auto_tokenizer = _RetryingAutoTokenizer()
    runner = HFModelRunner("google/gemma-4-E2B-it")

    tokenizer = runner._load_tokenizer(auto_tokenizer)

    assert isinstance(tokenizer, _DummyTokenizer)
    assert auto_tokenizer.calls == [
        ("google/gemma-4-E2B-it", {}),
        ("google/gemma-4-E2B-it", {"extra_special_tokens": {}}),
    ]
