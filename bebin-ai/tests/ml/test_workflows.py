from pathlib import Path

from ml.workflows.training_run import TrainingRunConfig, run_training_pipeline


def test_training_workflow_creates_reusable_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "sample.txt").write_text(
        "\n".join(
            [
                "Bebin AI trains from local text data and writes reusable artifacts.",
                "The workflow prepares data, trains a tokenizer, trains a model, and evaluates it.",
                "Repeatable training runs make local model improvement easier to inspect.",
            ]
        ),
        encoding="utf-8",
    )

    result = run_training_pipeline(
        TrainingRunConfig(
            raw_inputs=(raw_dir,),
            output_dir=tmp_path / "artifacts",
            min_chars=20,
            vocab_size=96,
            min_frequency=1,
            context_length=8,
            layers=1,
            heads=2,
            dim=16,
            dropout=0.0,
            batch_size=1,
            epochs=1,
            learning_rate=1e-3,
            validation_ratio=0.0,
            checkpoint_every_steps=1,
            use_mixed_precision=False,
            eval_max_new_tokens=1,
        )
    )

    assert result.processed_dataset_path.exists()
    assert result.tokenizer_path.exists()
    assert result.last_checkpoint_path.exists()
    assert result.best_checkpoint_path.exists()
    assert result.training_log_path.exists()
    assert result.manifest_path.exists()
    assert result.evaluation_report_path is not None
    assert result.evaluation_report_path.exists()
