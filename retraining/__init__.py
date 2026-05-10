"""Retraining pipeline entrypoints."""

INSIGHTFACE_REF = "f8613d444c6c266e8ff2fb29676a0a1cba6ee7a1"

__all__ = ["INSIGHTFACE_REF", "SCRFDRetrainingPipeline", "main", "make_training_version", "trigger_retraining"]


def __getattr__(name: str):
    if name in {"SCRFDRetrainingPipeline", "make_training_version", "trigger_retraining"}:
        if __package__:
            from . import pipeline
        else:
            import pipeline
        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_project_env() -> None:
    from pathlib import Path

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            import os
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _check_retraining_dependencies() -> None:
    missing = []
    for module in ["clearml", "optuna", "cv2", "azure.storage.blob", "mmcv", "torch", "onnx"]:
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)
    if missing:
        raise SystemExit(
            "Retraining dependencies are missing from the active environment:\n"
            f"  {', '.join(missing)}\n\n"
            "Activate the conda retraining environment, then rerun:\n"
            "  conda activate anonifyme-retraining\n"
            "  python -m retraining"
        )


def _check_clearml_configured() -> None:
    try:
        from clearml.backend_api.session.session import Session
        Session()
    except Exception as exc:
        if exc.__class__.__name__ != "MissingConfigError":
            raise
        raise SystemExit(
            "ClearML is not configured on this machine.\n\n"
            "Run:\n"
            "  clearml-init\n\n"
            "Or set these environment variables before retraining:\n"
            "  CLEARML_API_HOST\n"
            "  CLEARML_WEB_HOST\n"
            "  CLEARML_FILES_HOST\n"
            "  CLEARML_API_ACCESS_KEY\n"
            "  CLEARML_API_SECRET_KEY"
        ) from exc


def main() -> None:
    """Run the retraining pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Trigger the SCRFD retraining pipeline.")
    parser.add_argument(
        "--insightface-ref",
        default=INSIGHTFACE_REF,
        help="InsightFace branch, tag, or commit to checkout.",
    )
    args = parser.parse_args()
    _load_project_env()
    _check_retraining_dependencies()
    _check_clearml_configured()
    if __package__:
        from .pipeline import trigger_retraining
    else:
        from pipeline import trigger_retraining
    trigger_retraining(insightface_ref=args.insightface_ref)


if __name__ == "__main__":
    main()
