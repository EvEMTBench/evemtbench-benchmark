"""Layer 1 — bytes & integrity. NO science here.

Resolves data roots from env/config (never hardcoded), reads raw ``result{sim_idx}.csv``
waveforms and ``settings.csv`` metadata, joins by ``sim_idx`` (never row order), verifies
schemas (fail-loud) and file integrity (sha256 + corrupted-partition detection), and adapts
the pre-built window memmaps. This layer is self-contained: it does not depend on the
sibling ``psp_helper`` production package.
"""
