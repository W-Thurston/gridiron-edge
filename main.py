"""Docker / script entrypoint: delegates to Gridiron Edge CLI."""

from gridiron_edge.cli import main

if __name__ == "__main__":
    main()
