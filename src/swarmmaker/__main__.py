"""Module entrypoint for `python -m swarmmaker`."""

from .cli import app
from .config import settings


def main() -> None:
    _ = settings
    app()


if __name__ == "__main__":
    main()
