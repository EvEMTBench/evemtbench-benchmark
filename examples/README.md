# Examples

Runnable bring-your-own-model (BYO) walkthroughs. All of them need the dataset mounted
(`$EVEMT_ROOT`) and must run on a compute node (`srun`), never the login node — see
[docs/quickstart.md](../docs/quickstart.md) for the environment.

| script | what it shows |
|---|---|
| [`byo_model.py`](byo_model.py) | Both supported BYO paths: **(A)** implement the `Baseline` protocol + `register_baseline()` + run through the official runner (auditable `result.json`/`run_record.json` artifacts, identical to the built-ins), and **(B)** `load_task → predict → evaluate` for a quick look. Trains via the **streaming** view API (`iter_batches`), so it scales to views too large to materialize. |

```bash
srun --partition=a100mig --gres=gpu:a100small:1 --time=30:00 \
    bash -lc 'PYTHONPATH=src python examples/byo_model.py --grid double_line'
```

Quick API tour (small views only — `load_arrays` guards against materializing the big `global`
views; stream those instead):

```python
from evemtbench import list_tasks, get_task, load_task, load_arrays, evaluate

list_tasks()                                          # 24 task ids, 3 difficulty tiers
td = get_task("fault_detection_local")                # frozen TaskDefinition (metrics, tier, ...)
task = load_task(td.task_id, "held_out", grid="double_line")
X, y, meta = load_arrays(task, "train")               # dense (n, l, c) float32 + labels + metadata
```

Rules for a valid submission (see [docs/submission.md](../docs/submission.md)): fit on `train`
(+ `val` for model selection only), **never** touch `test`/`benchmark` during training or tuning,
report ≥3 seeds, and include the emitted `run_record.json`.
