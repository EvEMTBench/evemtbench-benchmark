"""``evemtbench-evaluate`` — score a predictions file against an official task/protocol.

This is what makes third-party results auditable: anyone can submit a predictions file and the
official metrics are recomputed here, independent of how the predictions were produced.

    evemtbench-evaluate --task fault_detection_local --protocol held_out \\
        --grid double_line --predictions preds.parquet --out result.json
"""

from __future__ import annotations

import argparse

from ..tasks.task import load_task
from .evaluate import evaluate
from .predictions_io import load_predictions


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a predictions file against a benchmark task.")
    p.add_argument("--task", default=None)
    p.add_argument("--protocol", default=None)
    p.add_argument("--grid", default=None)
    p.add_argument(
        "--predictions", default=None, help="parquet/csv: sample_id,window_idx,prediction"
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="verify the scoring harness itself (ceiling/floor/alignment controls; no data "
        "needed) and exit",
    )
    p.add_argument("--out", default=None, help="write the EvalResult JSON here (else stdout)")
    p.add_argument("--root", default=None)
    p.add_argument("--splits-root", default=None)
    p.add_argument("--directory", default=None)
    args = p.parse_args(argv)

    if args.self_test:
        from .selftest import main as selftest_main

        raise SystemExit(selftest_main())
    for req in ("task", "protocol", "predictions"):
        if getattr(args, req) is None:
            p.error(f"--{req} is required (unless --self-test)")

    task = load_task(
        args.task,
        args.protocol,
        grid=args.grid,
        root=args.root,
        splits_root=args.splits_root,
        directory=args.directory,
    )
    preds = load_predictions(args.predictions)
    result = evaluate(task, preds)

    text = result.to_json()
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
