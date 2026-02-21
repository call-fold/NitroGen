"""
Build script for packaging NitroGen test games as executables.

This allows the games to be controlled by NitroGen's play.py script,
which requires Windows executables with process names.

Usage:
    python test_game/build_game.py --game platformer
    python test_game/build_game.py --game simple
"""

import argparse
import subprocess
import sys
from pathlib import Path


def build_game(game_name, output_dir="dist"):
    """Build a game as an executable using PyInstaller."""

    games = {
        "platformer": {
            "source": "test_game/platformer_game.py",
            "name": "NitroGenPlatformerGame",
            "icon": None,  # Add path to .ico file if available
        },
        "simple": {
            "source": "test_game/simple_game.py",
            "name": "NitroGenSimpleGame",
            "icon": None,
        },
    }

    if game_name not in games:
        print(f"Unknown game: {game_name}")
        print(f"Available games: {', '.join(games.keys())}")
        return False

    game_info = games[game_name]
    source_path = Path(__file__).parent.parent / game_info["source"]

    if not source_path.exists():
        print(f"Source file not found: {source_path}")
        return False

    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name", game_info["name"],
        "--distpath", output_dir,
        "--workpath", str(Path(output_dir) / "build"),
        "--specpath", str(Path(output_dir) / "specs"),
        str(source_path),
    ]

    # Add icon if available
    if game_info["icon"]:
        cmd.extend(["--icon", game_info["icon"]])

    print("=" * 60)
    print(f"Building {game_info['name']}...")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Output: {output_dir}/{game_info['name']}.exe")
    print("")

    try:
        # Run PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=False)

        print("")
        print("=" * 60)
        print("Build completed successfully!")
        print("=" * 60)
        print(f"Executable: {Path(output_dir) / game_info['name']}.exe")
        print("")
        print("To run with NitroGen:")
        print(f"  1. Start the game: {output_dir}/{game_info['name']}.exe")
        print(f"  2. Start the server: python scripts/serve.py <model.pt> --port 5555")
        print(f"  3. Run the agent: python scripts/play.py --process \"{game_info['name']}.exe\" --port 5555")
        print("=" * 60)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Build failed with error: {e}")
        return False
    except FileNotFoundError:
        print("PyInstaller not found. Install it with:")
        print("  pip install pyinstaller")
        return False


def main():
    parser = argparse.ArgumentParser(description="Build NitroGen test games as executables")
    parser.add_argument(
        "--game",
        choices=["platformer", "simple"],
        default="platformer",
        help="Which game to build (default: platformer)"
    )
    parser.add_argument(
        "--output",
        default="dist",
        help="Output directory for the executable (default: dist)"
    )

    args = parser.parse_args()

    success = build_game(args.game, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
