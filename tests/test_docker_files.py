"""The image must contain every file the service reads at start. The first clean run of the final state failed because a new
config file (config/assumptions.yaml) was not copied: this test closes that class of mistakes."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
DOCKERIGNORE = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def copied_config_files() -> set[str]:
    out = set()
    for m in re.finditer(r"^COPY\s+(.+?)\s+(\S+)\s*$", DOCKERFILE, re.M):
        sources, dest = m.group(1).split(), m.group(2)
        for src in sources:
            if src.startswith("config/"):
                out.add(src.removeprefix("config/"))
        if sources == ["config"] or sources[0] == "config/":
            out |= {p.name for p in (ROOT / "config").iterdir()}
    return out


def test_every_config_file_is_copied_into_the_image():
    present = {p.name for p in (ROOT / "config").iterdir() if p.is_file()}
    missing = present - copied_config_files()
    assert not missing, f"в образ не копируются: {sorted(missing)}"


def test_the_dockerignore_does_not_exclude_what_the_image_needs():
    ignored = {l.strip().rstrip("/") for l in DOCKERIGNORE if l.strip() and not l.startswith("#") and not l.startswith("!")}
    for needed in ("config", "pipeline", "scripts", "web", "constraints.txt", "requirements-runtime.txt"):
        assert needed not in ignored, needed
    assert "*.md" in ignored and not any("xlsx" in l for l in ignored)                 # the workbook must reach the build


def test_the_runtime_modules_read_only_files_that_the_image_has():
    """Every path under config/ that the package reads must be one of the copied files."""
    reads = set()
    for p in (ROOT / "pipeline" / "analog_search").rglob("*.py"):
        reads |= set(re.findall(r'"config"\s*/\s*"([\w.]+)"', p.read_text(encoding="utf-8")))
    assert reads and reads <= copied_config_files(), sorted(reads - copied_config_files())
