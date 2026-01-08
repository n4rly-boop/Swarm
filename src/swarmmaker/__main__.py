"""Module entrypoint for `python -m swarmmaker`."""

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
