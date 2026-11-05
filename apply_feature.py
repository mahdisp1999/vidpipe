#!/usr/bin/env python3
"""Apply the Vidpipe multi-range stream-copy CLI integration.

Run this script from the Vidpipe repository root after extracting the ZIP there.
"""
from pathlib import Path

main = Path("vidpipe/main.py")
if not main.is_file() or not Path("vidpipe/cut.py").is_file():
    raise SystemExit("Extract this archive into the Vidpipe repository root first.")

text = main.read_text(encoding="utf-8")

def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"Safe patch failed for {label}: expected one anchor, found {count}.")
    return source.replace(old, new, 1)

if "from .cut import remove_ranges" not in text:
    text = replace_once(
        text,
        "from .controller import Controller, State, StateError\n",
        "from .controller import Controller, State, StateError\nfrom .cut import remove_ranges\n",
        "imports",
    )

if '"--remove-range"' not in text:
    text = replace_once(
        text,
        '    parser.add_argument("input", nargs="?", help="input video file (FFmpeg detects its format by default)")\n',
        '    parser.add_argument("input", nargs="?", help="input video file (FFmpeg detects its format by default)")\n'
        '    parser.add_argument(\n'
        '        "--remove-range", action="append", metavar="START-END",\n'
        '        help="remove a time range without re-encoding; repeat for multiple ranges",\n'
        '    )\n',
        "CLI argument",
    )

if "return remove_ranges(args.input" not in text:
    text = replace_once(
        text,
        "    try:\n        return _run(args)\n",
        "    try:\n"
        "        if args.remove_range:\n"
        "            return remove_ranges(args.input, args.remove_range, args.output, args.overwrite)\n"
        "        return _run(args)\n",
        "main dispatch",
    )

main.write_text(text, encoding="utf-8")
print("Vidpipe CLI integration applied.")
print("Run: python -m pytest -q")
print("Example: vidpipe input.ts --remove-range 00:10:00-00:15:00 --remove-range 00:40:00-00:50:00")
