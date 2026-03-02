#!/usr/bin/env python3
"""Audit LaTeX citations vs. a BibTeX/BibLaTeX .bib file.

Default behavior:
- Scans .tex files under _Jayme_Final_Project (excluding Chapter-Example)
- Extracts citation keys from common biblatex commands (\cite*, \textcite, \parencite,
  \autocite, \footcite, \citeauthor, \citeyear, \nocite, \apud, etc.)
- Compares against keys defined in Post-Textual/references.bib
- Reports missing cited keys and unused bibliography entries

Optionally writes a pruned .bib containing only cited entries.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


_TEX_CMD_START = re.compile(r"\\([A-Za-z@]+)\*?")
_BIB_ENTRY_START = re.compile(r"@([A-Za-z]+)\s*([({])", re.MULTILINE)
_KEY_TOKEN = re.compile(r"[A-Za-z0-9_.:-]+")


def _strip_tex_comment(line: str) -> str:
    out = []
    escaped = False
    for ch in line:
        if ch == "\\":
            escaped = not escaped
            out.append(ch)
            continue
        if ch == "%" and not escaped:
            break
        escaped = False
        out.append(ch)
    return "".join(out)


def _iter_tex_files(root: Path, exclude_globs: list[str]) -> Iterator[Path]:
    for p in root.rglob("*.tex"):
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in exclude_globs):
            continue
        yield p


def _extract_cite_keys_from_brace_group(group: str) -> set[str]:
    keys: set[str] = set()
    for part in group.split(","):
        part = part.strip()
        if not part or part == "*":
            continue
        m = _KEY_TOKEN.search(part)
        if m:
            keys.add(m.group(0))
    return keys


def _read_balanced(text: str, i: int, open_ch: str, close_ch: str) -> tuple[str, int] | None:
    if i >= len(text) or text[i] != open_ch:
        return None
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
        j += 1
    return None


def extract_cite_keys_from_tex(tex: str) -> set[str]:
    keys: set[str] = set()
    i = 0
    while i < len(tex):
        if tex[i] != "\\":
            i += 1
            continue

        m = _TEX_CMD_START.match(tex, i)
        if not m:
            i += 1
            continue

        cmd = m.group(1)
        cmd_l = cmd.lower()
        i = m.end()

        # Only consider citation-like commands.
        if "cite" not in cmd_l and cmd_l not in {"nocite", "apud"}:
            continue

        # Skip whitespace
        while i < len(tex) and tex[i].isspace():
            i += 1

        # Consume any number of optional arguments [...]
        while True:
            opt = _read_balanced(tex, i, "[", "]")
            if not opt:
                break
            _, i = opt
            while i < len(tex) and tex[i].isspace():
                i += 1

        # Consume one or more mandatory arguments {...}
        consumed_any = False
        while True:
            grp = _read_balanced(tex, i, "{", "}")
            if not grp:
                break
            consumed_any = True
            content, i = grp
            keys |= _extract_cite_keys_from_brace_group(content)
            while i < len(tex) and tex[i].isspace():
                i += 1

        if not consumed_any:
            # Avoid infinite loops on weird constructs.
            i = m.end()

    return keys


def extract_cite_keys_from_tex_file(path: Path) -> set[str]:
    in_verbatim = False
    begin_env = re.compile(r"\\begin\{(verbatim|Verbatim|lstlisting|minted)\}")
    end_env = re.compile(r"\\end\{(verbatim|Verbatim|lstlisting|minted)\}")

    buf: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines(True):
        line = _strip_tex_comment(raw_line)
        if not in_verbatim and begin_env.search(line):
            in_verbatim = True
            continue
        if in_verbatim:
            if end_env.search(line):
                in_verbatim = False
            continue
        buf.append(line)

    return extract_cite_keys_from_tex("".join(buf))


@dataclass(frozen=True)
class BibSegment:
    kind: str  # "text" or "entry"
    text: str
    entry_type: str | None = None
    key: str | None = None


def _find_bib_entry_end(text: str, start_match: re.Match[str]) -> int:
    open_ch = start_match.group(2)
    close_ch = ")" if open_ch == "(" else "}"

    i = start_match.end()
    depth = 1
    while i < len(text):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("Unterminated .bib entry")


def parse_bib_segments(text: str) -> list[BibSegment]:
    segs: list[BibSegment] = []
    pos = 0
    for m in _BIB_ENTRY_START.finditer(text):
        start = m.start()
        if start > pos:
            segs.append(BibSegment(kind="text", text=text[pos:start]))

        end = _find_bib_entry_end(text, m)
        raw = text[start:end]
        entry_type = m.group(1).lower()

        key: str | None = None
        if entry_type not in {"preamble", "string", "comment"}:
            j = m.end()
            while j < len(text) and text[j].isspace():
                j += 1
            k = j
            while k < len(text) and text[k] not in {",", "\n", "\r"}:
                k += 1
            key = text[j:k].strip() or None

        segs.append(BibSegment(kind="entry", text=raw, entry_type=entry_type, key=key))
        pos = end

    if pos < len(text):
        segs.append(BibSegment(kind="text", text=text[pos:]))

    return segs


def bib_keys(segs: Iterable[BibSegment]) -> set[str]:
    return {s.key for s in segs if s.kind == "entry" and s.key}


def _closure_keep_keys(segs: Iterable[BibSegment], initial: set[str]) -> set[str]:
    keep = set(initial)
    changed = True

    crossref_re = re.compile(r"\\b(crossref|xdata)\\s*=\\s*\{([^}]+)\}", re.IGNORECASE)
    while changed:
        changed = False
        for s in segs:
            if s.kind != "entry" or not s.key or s.key not in keep:
                continue
            for _, vals in crossref_re.findall(s.text):
                for k in _extract_cite_keys_from_brace_group(vals):
                    if k not in keep:
                        keep.add(k)
                        changed = True

    return keep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-root", default="_Jayme_Final_Project", help="Root folder to scan for .tex")
    ap.add_argument(
        "--exclude",
        action="append",
        default=["Chapter-Example/**"],
        help="Glob (relative to tex-root) to exclude; can be repeated",
    )
    ap.add_argument(
        "--bib",
        default="_Jayme_Final_Project/Post-Textual/references.bib",
        help=".bib to validate/prune",
    )
    ap.add_argument("--list-missing", action="store_true", help="Print missing cited keys")
    ap.add_argument("--list-unused", action="store_true", help="Print unused .bib entry keys")
    ap.add_argument("--prune-out", help="Write pruned .bib to this path")
    ap.add_argument("--in-place", action="store_true", help="Overwrite --bib (creates .bak)")

    ns = ap.parse_args(argv)

    tex_root = Path(ns.tex_root)
    bib_path = Path(ns.bib)
    exclude = list(ns.exclude)

    cited: set[str] = set()
    for p in _iter_tex_files(tex_root, exclude):
        cited |= extract_cite_keys_from_tex_file(p)

    bib_text = bib_path.read_text(encoding="utf-8", errors="ignore")
    segs = parse_bib_segments(bib_text)
    defined = bib_keys(segs)

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)

    print(f"TEX cited keys: {len(cited)}")
    print(f"BIB defined keys: {len(defined)} ({bib_path})")
    print(f"Missing (cited but not in BIB): {len(missing)}")
    print(f"Unused (in BIB but never cited): {len(unused)}")

    if ns.list_missing and missing:
        print("\n# Missing keys")
        print("\n".join(missing))

    if ns.list_unused and unused:
        print("\n# Unused keys")
        print("\n".join(unused))

    if ns.prune_out or ns.in_place:
        keep = _closure_keep_keys(segs, cited & defined)
        out_text = "".join(
            s.text
            for s in segs
            if s.kind == "text" or s.key is None or (s.kind == "entry" and s.key in keep)
        )

        if ns.in_place:
            bak = bib_path.with_suffix(bib_path.suffix + ".bak")
            bak.write_text(bib_text, encoding="utf-8")
            bib_path.write_text(out_text, encoding="utf-8")
            print(f"\nWrote pruned bibliography in-place: {bib_path} (backup: {bak.name})")
        else:
            out_path = Path(ns.prune_out)
            out_path.write_text(out_text, encoding="utf-8")
            print(f"\nWrote pruned bibliography: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
