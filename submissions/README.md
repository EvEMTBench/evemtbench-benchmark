# Submissions

Outside results for the EvEMTBench leaderboard, one directory per model. Open a pull request that
adds `submissions/<model>/`. CI runs the validator on every directory here, and the leaderboard
lists valid submissions next to the reference baselines with a tier badge:

- **self-reported**: the files pass the validator; the numbers are the submitter's.
- **reproduced**: a maintainer re-ran the submitted code and got the submitted results.

The format, the metadata fields and the rules (validation-only model selection, ≥ 3 seeds, no
test or benchmark data in training) are in [docs/submission.md](../docs/submission.md#6-submitting-to-the-leaderboard).
Check a submission locally before opening the pull request:

```bash
evemtbench-validate-submission submissions/<model>
```
