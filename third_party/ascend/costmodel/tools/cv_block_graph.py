#!/usr/bin/env python3
"""Turn the CV-pipeline cost report into a picture of the compute blocks.

The pass prints one line per block (its core, its cost, what it contains) and
one line per dependency. That is enough to draw the thing the CV pipeline
actually built: which blocks run on which core, what each costs, and which of
them have to wait for each other.

Usage:
    TRITON_ASCEND_CV_COST_VERBOSE=2 python AOTcompile.py 2> cost.log
    python cv_block_graph.py cost.log --format dot > blocks.dot
    dot -Tsvg blocks.dot -o blocks.svg

    python cv_block_graph.py cost.log            # text summary, no graphviz

Reads stdin when no file is given.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

PREFIX = r"\[estimate-cv-pipeline-cost\]"

# "      12  CUBE       3288       0      3288    1    2  -      fixpipe_ub   ..."
#   id  core  cycles  start  finish  seg  ops  waits  unit  contents
BLOCK_RE = re.compile(
    PREFIX + r"\s+(?P<id>-{2}|\d+)\s+(?P<core>CUBE|VECTOR|MIXED)\s+"
    r"(?P<cycles>\d+)\s+(?P<start>\d+)\s+(?P<finish>\d+)\s+"
    r"(?P<segments>\d+)\s+(?P<ops>\d+)\s+(?P<waits>\S+)\s+(?P<unit>\S+)"
    r"\s+(?P<contents>.*?)\s*$"
)
# "    block 7 <- 12, 3"
DEP_RE = re.compile(PREFIX + r"\s+block\s+(?P<id>\d+)\s+<-\s+(?P<producers>.*?)\s*$")
# "block sync (waits on a flag the other block sets):"
DEP_KIND_RE = re.compile(PREFIX + r"\s+block\s+(?P<kind>sync|dataflow)\s+\(")
# "Ascend custom (...): 62250 cycles (33.6 us) over 14 block(s), 41500 cycles
#  if fully overlapped, ..."
TOTAL_RE = re.compile(
    PREFIX + r"\s+(?P<hardware>.+?):\s+(?P<cycles>\d+)\s+cycles"
    r"\s+\((?P<us>[\d.]+)\s+us\)\s+over\s+\d+\s+block\(s\),"
    r"\s+(?P<roofline>\d+)\s+cycles if fully overlapped"
)


@dataclass
class Block:
    id: int
    core: str
    cycles: int
    ops: int
    unit: str
    contents: str
    # Position in the one-iteration schedule, and what held the block up:
    # a block id, "core" (its own core was still busy) or "-" (nothing).
    start: int = 0
    finish: int = 0
    segments: int = 1
    waits: str = "-"
    sync_on: list[int] = field(default_factory=list)
    reads_from: list[int] = field(default_factory=list)

    @property
    def name(self) -> str:
        return "unassigned" if self.id < 0 else f"block {self.id}"

    @property
    def node_id(self) -> str:
        # Graphviz ids may not contain '-', so the unassigned bucket needs a
        # name of its own rather than "b-1".
        return "b_unassigned" if self.id < 0 else f"b{self.id}"


@dataclass
class Report:
    blocks: dict[int, Block] = field(default_factory=dict)
    hardware: str = "?"
    total_cycles: int = 0
    total_us: float = 0.0
    # The same module with Cube and Vector assumed to overlap unconditionally.
    # The gap to total_cycles is what the barriers cost.
    roofline_cycles: int = 0


def parse(lines) -> Report:
    report = Report()
    # The two dependency sections are introduced by a header line; edges
    # after it belong to that kind until the next header.
    kind = "sync"
    for line in lines:
        if (m := DEP_KIND_RE.search(line)) is not None:
            kind = m.group("kind")
            continue
        if (m := TOTAL_RE.search(line)) is not None:
            report.hardware = m.group("hardware")
            report.total_cycles = int(m.group("cycles"))
            report.total_us = float(m.group("us"))
            report.roofline_cycles = int(m.group("roofline"))
            continue
        if (m := BLOCK_RE.search(line)) is not None:
            raw_id = m.group("id")
            block_id = -1 if raw_id == "--" else int(raw_id)
            report.blocks[block_id] = Block(
                id=block_id,
                core=m.group("core"),
                cycles=int(m.group("cycles")),
                ops=int(m.group("ops")),
                unit=m.group("unit"),
                contents=m.group("contents"),
                start=int(m.group("start")),
                finish=int(m.group("finish")),
                segments=int(m.group("segments")),
                waits=m.group("waits"),
            )
            continue
        if (m := DEP_RE.search(line)) is not None:
            block_id = int(m.group("id"))
            producers = [int(p) for p in re.findall(r"-?\d+", m.group("producers"))]
            if block_id in report.blocks:
                target = report.blocks[block_id]
                if kind == "sync":
                    target.sync_on = producers
                else:
                    target.reads_from = producers
    return report


# Blocks on the same core serialise, so the core totals are what the estimate
# is actually bounded by. Blocks on different cores overlap, which is why these
# do not add up to the module total.
def core_totals(report: Report) -> dict[str, int]:
    totals: dict[str, int] = {}
    for block in report.blocks.values():
        totals[block.core] = totals.get(block.core, 0) + block.cycles
    return totals


def emit_text(report: Report, out) -> None:
    if not report.blocks:
        print("no blocks found -- was the log produced with "
              "TRITON_ASCEND_CV_COST_VERBOSE=2?", file=sys.stderr)
        return

    ordered = sorted(report.blocks.values(), key=lambda b: -b.cycles)
    widest = max(b.cycles for b in ordered) or 1

    print(f"hardware: {report.hardware}", file=out)
    print(f"module estimate: {report.total_cycles} cycles "
          f"({report.total_us:.3f} us)", file=out)
    if report.roofline_cycles:
        penalty = report.total_cycles - report.roofline_cycles
        share = 100.0 * penalty / report.roofline_cycles
        print(f"if Cube and Vector overlapped freely: "
              f"{report.roofline_cycles} cycles "
              f"(barriers cost {penalty}, +{share:.1f}%)", file=out)
    split = [b for b in report.blocks.values() if b.segments > 1]
    if split:
        ids = ", ".join(b.name for b in sorted(split, key=lambda b: b.id))
        print(f"{len(split)} block(s) with a barrier inside them: {ids}",
              file=out)
    print(file=out)

    print(f"{'block':>11}  {'core':<7}{'cycles':>12}  {'waits':<7}"
          f"{'bottleneck':<12} {'':<22} contents", file=out)
    for block in ordered:
        bar = "#" * max(1, round(20 * block.cycles / widest))
        print(f"{block.name:>11}  {block.core:<7}{block.cycles:>12}  "
              f"{block.waits:<7}{block.unit:<12} {bar:<22} {block.contents}",
              file=out)

    print("\nper-core totals (blocks on one core run one after another):",
          file=out)
    for core, total in sorted(core_totals(report).items(), key=lambda kv: -kv[1]):
        print(f"  {core:<7}{total:>12} cycles", file=out)

    sync = sorted((b.id, p) for b in ordered for p in b.sync_on)
    if sync:
        print("\nsync edges (consumer waits on a flag the producer sets).",
              file=out)
        print("These are the hard barriers: the consumer cannot start before "
              "the producer\nfinishes, which is what the estimate charges for "
              "on top of the roofline:", file=out)
        for consumer, producer in sync:
            print(f"  block {consumer} <- block {producer}", file=out)

    data = sorted((b.id, p) for b in ordered for p in b.reads_from)
    if data:
        print("\ndataflow edges (consumer reads a value the producer made):",
              file=out)
        for consumer, producer in data:
            print(f"  block {consumer} <- block {producer}", file=out)


def emit_dot(report: Report, out) -> None:
    widest = max((b.cycles for b in report.blocks.values()), default=1) or 1
    fills = {"CUBE": "#cfe3f7", "VECTOR": "#ffe4c4", "MIXED": "#f2c9c9"}

    print("digraph cv_blocks {", file=out)
    print('  rankdir=TB;', file=out)
    print('  node [shape=box, style="filled,rounded", fontname="Helvetica", '
          'fontsize=10];', file=out)
    print('  edge [color="#666666"];', file=out)
    print(f'  label="{report.hardware}\\n{report.total_cycles} cycles '
          f'({report.total_us:.3f} us)"; labelloc=t; fontsize=12;', file=out)

    # Group by core so the two pipelines read as two columns.
    for core in ("CUBE", "VECTOR", "MIXED"):
        members = [b for b in report.blocks.values() if b.core == core]
        if not members:
            continue
        print(f'  subgraph cluster_{core.lower()} {{', file=out)
        print(f'    label="{core}"; style=dashed; color="#999999";', file=out)
        for block in sorted(members, key=lambda b: b.id):
            share = block.cycles / widest
            # Thicker border the more the block costs, so the hot ones stand out.
            width = 1 + round(3 * share)
            contents = block.contents.replace('"', r"\"")
            label = (f"{block.name}\\n{block.cycles} cycles\\n"
                     f"{block.unit} | {block.ops} ops\\n{contents}")
            print(f'    {block.node_id} [label="{label}", '
                  f'fillcolor="{fills[core]}", penwidth={width}];', file=out)
        print("  }", file=out)

    # Solid red for a flag wait, dashed grey for plain dataflow: the first
    # kind stops a core, the second only orders work already on one.
    for block in report.blocks.values():
        for producer in block.sync_on:
            if producer in report.blocks:
                print(f'  {report.blocks[producer].node_id} -> '
                      f'{block.node_id} [color="#c0392b", penwidth=2, '
                      f'label="sync", fontsize=8];', file=out)
        for producer in block.reads_from:
            if producer in report.blocks:
                print(f'  {report.blocks[producer].node_id} -> '
                      f'{block.node_id} [style=dashed];', file=out)
    print("}", file=out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="cost report; stdin if omitted")
    parser.add_argument("--format", choices=("text", "dot"), default="text")
    args = parser.parse_args()

    source = open(args.log, encoding="utf-8") if args.log else sys.stdin
    with source:
        report = parse(source)

    if not report.blocks:
        print("no per-block lines found; run with "
              "TRITON_ASCEND_CV_COST_VERBOSE=2 and capture stderr",
              file=sys.stderr)
        return 1

    (emit_dot if args.format == "dot" else emit_text)(report, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
