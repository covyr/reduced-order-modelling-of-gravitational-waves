from pathlib import Path

def empty_directory(directory: Path) -> None:
    """Removes all files from a directory."""
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()
