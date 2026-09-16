"""The animation calls users copy from must use cleopatra's current keywords.

cleopatra 0.38 moved `interval`, `frame_label`, `cell_value_text_colors` and `data_getter` from
loose keywords of `ArrayGlyph.animate` into `playback=Animation(...)`, and it raises on a keyword it
no longer knows rather than ignoring it. `SimulationResults.animate` forwards its keywords
untouched, so a documented call that still passes one loose fails the moment a reader runs it --
and nothing in the docs build executes the snippet to notice.

This parses the Python in the docs' code blocks, the example scripts and the test drivers, and
fails on any `.animate(...)` call that passes one of those keywords directly.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Keywords cleopatra 0.38 moved into `playback=Animation(...)`.
PLAYBACK_KEYWORDS = frozenset(
    {"interval", "frame_label", "cell_value_text_colors", "data_getter"}
)

CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def sources() -> list[tuple[str, str]]:
    """Return `(label, python source)` for every snippet and script a user might run."""
    found: list[tuple[str, str]] = []
    for page in sorted((REPO / "docs").rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        for index, block in enumerate(CODE_BLOCK.findall(text)):
            found.append((f"{page.relative_to(REPO).as_posix()}[block {index}]", block))
    for tree in ("examples", "tests/scripts"):
        for script in sorted((REPO / tree).rglob("*.py")):
            label = script.relative_to(REPO).as_posix()
            found.append((label, script.read_text(encoding="utf-8", errors="replace")))
    return found


def loose_playback_keywords() -> tuple[list[str], int]:
    """Return the offending calls and how many sources parsed at all."""
    offending: list[str] = []
    parsed = 0
    for label, source in sources():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        parsed += 1
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "animate"
            ):
                continue
            loose = sorted({kw.arg for kw in node.keywords} & PLAYBACK_KEYWORDS)
            if loose:
                offending.append(f"{label}:{node.lineno} passes {loose} to animate()")
    return offending, parsed


def test_no_documented_animate_call_passes_a_playback_keyword_loose():
    """Test that docs, examples and drivers pass playback options through `Animation`.

    Test scenario:
        The distributed-calibration page documented `animate(..., interval=200,
        frame_label=FrameLabel(...))`, which cleopatra 0.39 rejects with `ValueError`, after
        every script had already been moved to `playback=Animation(...)`.
    """
    offending, parsed = loose_playback_keywords()

    assert parsed > 20, (
        f"only {parsed} sources parsed, so the scan is measuring too little"
    )
    assert not offending, (
        "use playback=Animation(...) for these keywords:\n" + "\n".join(offending)
    )
