# data/

## Source: data/combined/run_16node.jsonl

The repo-root file `data/combined/run_16node.jsonl` records one 16-node rack.
It's in **wide format**: each line is one polling interval (`index`) for the
whole rack, not one node. Every line has:

- `index`, `FRQ` -- shared across all 16 nodes (one rack-level ENF/PDU
  reading per interval, there's no per-node FRQ).
- 16 blocks of columns, one block per node, each key prefixed with that
  node's hostname, e.g.:

  ```
  x3105c0s37b0n0_gpu-0[W]
  x3105c0s37b0n0_gpu-0[C]
  x3105c0s37b0n0_cpu-0[uJ]
  x3105c0s37b0n0_cpu-0-core[uJ]
  x3105c0s37b0n0_cpu-0[W]
  x3105c0s37b0n0_cpu-0-core[W]
  ...
  ```

So there is no row range that "belongs" to a node -- all 16 nodes' readings
for a given interval live on the same line. A node's time series is a
*column slice* (every key starting with its hostname, across all 1800
lines), not a *row slice*.

This doesn't match what `models.py`'s `TelemetrySample.from_dict` parses:
its `_GPU_KEY_RE`/`_CPU_KEY_RE` regexes expect keys like `gpu-0[W]` and
`cpu-0[uJ]` with no hostname prefix (the shape `data/run01.jsonl` is
already in), so pointed at `run_16node.jsonl` directly it would silently
produce empty `gpu_power_w`/`cpu_power_w` dicts for every sample.

## How the per-node files here were produced

`tools/split_16node.py` regroups `run_16node.jsonl` by hostname: for each
line, it splits the 16 column blocks apart by their `<hostname>_` prefix,
strips the prefix, and re-attaches the shared `index`/`FRQ` to each node's
record. The result is one `node00.jsonl`..`node15.jsonl` file per node
(numbered by sorting the source hostnames alphabetically, so numbering is
stable across regeneration), in the same `index`/`FRQ`/`gpu-N[...]`/
`cpu-N[...]` shape as `run01.jsonl`, so `data_feed.py`'s existing
single-node loader (via `TelemetrySample`) reads any of them unchanged --
no parser changes needed. The original hostname per node is only printed
to the terminal when regenerating, not stored in the output files.

Regenerate with:

```
python lanes/mccray_dashboard/tools/split_16node.py
```

By default it reads `data/combined/run_16node.jsonl` (repo root) and writes
into this directory; pass `--input`/`--output-dir` to override.

## Files

- `run01.jsonl` -- original single-node recording used by the dashboard's
  first version.
- `node00.jsonl`..`node15.jsonl` (16 files) -- one per node from
  `run_16node.jsonl`, produced by `split_16node.py` as described above.
  `data_feed.py` derives a node's id from the filename stem (see
  `get_rack_id()`), so loading e.g. `node07.jsonl` gives that node the id
  `node07`, displayed in the dashboard as "Node 07".
