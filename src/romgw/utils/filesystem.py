from pathlib import Path

def empty_directory(directory: str | Path) -> None:
    """Removes all files from a directory."""
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
