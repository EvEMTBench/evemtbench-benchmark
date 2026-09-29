# Data License

The **code** in this repository is licensed under MIT (see [`/LICENSE`](../LICENSE)).

The **EvEMTBench dataset** (waveform recordings, labels, metadata, topology graphs) is a
**separate artifact** distributed under its own terms and is **not** contained in this repository.

## Dataset license: CC-BY-4.0 (intended)

The EvEMTBench dataset is released separately (FAUDataCloud; see the README's *Dataset* section) under the
**Creative Commons Attribution 4.0 International (CC-BY-4.0)** license, unless the dataset's
own landing page states otherwise. When you use the dataset, cite it per the dataset DOI and
this repository's [`CITATION.cff`](../CITATION.cff).

## PowerFactory / simulation-tool note

The dataset was generated with **DIgSILENT PowerFactory**, a proprietary EMT simulation tool.

- The **released waveform exports and derived windows** are data products and may be used and
  redistributed under the dataset license above. The redistribution terms for the PowerFactory
  **simulation outputs** have been confirmed.
- **Regeneration** of the raw data from the PowerFactory models may require a PowerFactory
  license. This benchmark does not redistribute PowerFactory or its model files.

## What this repository does contain

- Source code (MIT).
- Committed **split files** (`src/evemtbench/splits/`) — lists of simulation IDs, derived from
  dataset metadata. These are released under MIT as code-adjacent artifacts.
- A tiny **synthetic fixture** (`tests/fixtures/`) — fabricated, not real grid data — for testing.
- **SHA-256 manifests** of dataset files (`src/evemtbench/data/manifests/`) — checksums only,
  no waveform content.
