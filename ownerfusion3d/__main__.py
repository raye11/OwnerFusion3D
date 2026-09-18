"""Allow ``python -m ownerfusion3d`` as the public unified entry point."""

from .cli.run import main


if __name__ == "__main__":
    main()
