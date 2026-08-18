"""Loader-stage probe (2026-07-17 GPU audit follow-up): WHERE does the ~58% data-wait go?

Times, on the production cell cigre_mv/event_detection_global (mlp view, bs=256):
  first-batch wait  -> DataLoader (re)creation cost: 8-worker fork + warm-up
                       (per epoch, x2 with val)
  steady batch wait -> per-batch pipeline: worker read -> IPC -> pin -> (H2D excluded)
across configs that discriminate the remaining hypotheses. Read-only; touches no repo state.

Env in: EVEMTBENCH_DATA_ROOT, EVEMTBENCH_PROJECTION_CACHE (+_ROOT), EVEMTBENCH_CACHE_READONLY=1.
Args: --cache-dir <dir> --tag <label> [--configs baseline,nopin,sync,persistent,pack4,pack8,raw]
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evemtbench.baselines._loader import _identity, prefetched_batches
from evemtbench.tasks.task import build_input_transform, load_task

BS = 256
EPOCH_BATCHES = None  # None = full pass; sync config overrides


def timed_pass(gen, limit=None):
    t0 = time.perf_counter()
    first, waits, nb, nbytes = None, [], 0, 0
    prev = t0
    for xb, _yb in gen:
        if isinstance(xb, list):  # packed: list of (x, y) tuples
            nb += len(xb)
            nbytes += sum(t[0].numel() * 4 for t in xb)
        else:
            nb += 1
            nbytes += xb.numel() * 4
        dt = time.perf_counter() - prev
        if first is None:
            first = dt
        else:
            waits.append(dt)
        prev = time.perf_counter()
        if limit and nb >= limit:
            break
    total = time.perf_counter() - t0
    return dict(
        first_s=first,
        steady_ms=1e3 * float(np.mean(waits)) if waits else float("nan"),
        p90_ms=1e3 * float(np.percentile(waits, 90)) if waits else float("nan"),
        nb=nb,
        total_s=total,
        gb_s=nbytes / total / 1e9,
        samp_s=nb * BS / total,
    )


class PackedDataset(Dataset):
    """k production batches per worker item — same batch sequence, k× fewer IPC round-trips."""

    def __init__(self, view, batches, transform, k):
        self.v, self.b, self.t, self.k = view, batches, transform, k

    def __len__(self):
        return (len(self.b) + self.k - 1) // self.k

    def __getitem__(self, i):
        out = []
        for j in range(i * self.k, min((i + 1) * self.k, len(self.b))):
            pos = self.b[j]
            xb = self.v._read_positions(pos)
            if self.t is not None:
                xb = self.t(xb)
            xb = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32))
            out.append((xb, self.v.labels[pos]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--configs", default="baseline,nopin,sync,persistent,pack4,pack8,raw")
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    os.environ["EVEMTBENCH_PROJECTION_CACHE"] = args.cache_dir

    task = load_task(
        "event_detection_global",
        "held_out",
        grid="cigre_mv",
        root=os.environ["EVEMTBENCH_DATA_ROOT"],
    )
    train = task.train
    t0 = time.perf_counter()
    train.materialize_cache(args.cache_dir)
    print(f"[{args.tag}] materialize_cache {time.perf_counter() - t0:.1f}s")
    transform = build_input_transform(False, "none")  # mlp defaults: zero_seq=0, normalize=none
    n = train.feature_shape[0]
    print(f"[{args.tag}] view n={n} shape={train.feature_shape} bs={BS}")
    order = np.random.default_rng(0).permutation(n)
    batches = [order[i : i + BS] for i in range(0, n, BS)]
    cfgs = args.configs.split(",")

    def report(name, p, r):
        print(
            f"RESULT tag={args.tag} cfg={name} pass={p} first_s={r['first_s']:.2f} "
            f"steady_ms={r['steady_ms']:.1f} p90_ms={r['p90_ms']:.1f} nb={r['nb']} "
            f"total_s={r['total_s']:.1f} gb_s={r['gb_s']:.2f} samp_s={r['samp_s']:.0f}",
            flush=True,
        )

    for p in range(args.passes):
        if "baseline" in cfgs:  # production path: fresh loader each pass, 8 workers, pinned
            r = timed_pass(
                prefetched_batches(
                    train, BS, order=order, transform=transform, workers=8, pin_memory=True
                )
            )
            report("baseline_w8_pin", p, r)
        if "nopin" in cfgs:
            r = timed_pass(
                prefetched_batches(
                    train, BS, order=order, transform=transform, workers=8, pin_memory=False
                )
            )
            report("w8_nopin", p, r)
        if "sync" in cfgs:  # single-process floor, 100 batches
            r = timed_pass(
                prefetched_batches(
                    train, BS, order=order, transform=transform, workers=0, pin_memory=False
                ),
                limit=100,
            )
            report("sync_w0", p, r)
        if "pack4" in cfgs or "pack8" in cfgs:
            for k in (4, 8):
                if f"pack{k}" not in cfgs:
                    continue
                dl = DataLoader(
                    PackedDataset(train, batches, transform, k),
                    batch_size=1,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True,
                    collate_fn=_identity,
                    prefetch_factor=4,
                    persistent_workers=False,
                )
                r = timed_pass(iter(dl))
                report(f"pack{k}_w8_pin", p, r)
    if "persistent" in cfgs:  # ONE loader, re-iterated: fork paid once, then warm re-iterations
        from evemtbench.baselines._loader import _PositionBatchDataset

        dl = DataLoader(
            _PositionBatchDataset(train, batches, transform),
            batch_size=1,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=_identity,
            prefetch_factor=4,
            persistent_workers=True,
        )
        for p in range(args.passes):
            report("persistent_w8_pin", p, timed_pass(iter(dl)))
        del dl
    if "raw" in cfgs:  # main-process gather floor (no DataLoader at all), 100 batches
        for p in range(args.passes):
            t = time.perf_counter()
            nb = 0
            for b in batches[:100]:
                xb = train._read_positions(b)
                if transform is not None:
                    xb = transform(xb)
                _ = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32))
                nb += 1
            total = time.perf_counter() - t
            print(
                f"RESULT tag={args.tag} cfg=raw_gather pass={p} first_s=0.00"
                f" steady_ms={1e3 * total / nb:.1f} p90_ms=nan nb={nb}"
                f" total_s={total:.1f} gb_s=nan samp_s={nb * BS / total:.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
