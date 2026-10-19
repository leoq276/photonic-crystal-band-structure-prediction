"""Check that bundled binary assets match the reviewed files."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for line in (ROOT / "ASSET_SHA256.txt").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().upper() != expected.upper():
            raise RuntimeError(f"Hash mismatch: {relative}")
        print(f"OK {relative}")


if __name__ == "__main__":
    main()
