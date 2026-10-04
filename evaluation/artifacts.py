"""Write evaluation outputs without replacing previous runs."""

from pathlib import Path


def unused_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def write_csv_new(frame, path: str | Path) -> Path:
    target = unused_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target
