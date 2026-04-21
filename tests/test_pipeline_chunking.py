from app.model_runner import MockModelRunner
from app.pipeline import run_pipeline


def test_pipeline_uses_chunking_for_long_input(tmp_path) -> None:
    lines = []
    for idx in range(1, 26):
        lines.append(f"Client: Requirement line {idx} should support booking action {idx}.")

    conversation_path = tmp_path / "long_conversation.txt"
    conversation_path.write_text("\n".join(lines), encoding="utf-8")

    output_dir = tmp_path / "output"
    runner = MockModelRunner()

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "ambiguous_phrases": ["simple", "easy to use"],
            "generation": {},
            "enable_chunking": True,
            "chunk_token_budget": 80,
            "max_chunk_utterances": 4,
            "enable_final_refinement": False,
        },
    )

    assert result.diagnostics.chunks_used > 1
    assert len(result.spec.utterances) == 25
    assert result.json_path.exists()
    assert result.markdown_path.exists()
