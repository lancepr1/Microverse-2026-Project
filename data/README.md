# Datasets

Datasets are large and access-controlled. They are gitignored and never
committed. Put them here locally with these paths so the loaders find them.

## NLR GenAI Workload Power Profiles (2026)

- Source: https://data.nlr.gov/submissions/312  (about 1 GB, README at the top)
- Paper: Vercellino et al. 2026
- Local path expected by the loader: `data/nlr/`
- Primary source for Hendricks. Secondary for Marchisano and McCray.
- Week 1: everyone at least skims the file structure. Hendricks confirms the
  real column schema and updates `load_nlr_profile` in `data_loaders.py`.

## 2025 ENF measurements (AFRL Rome)

- Source: from the lab / Dr. Qu, read access granted in week 1.
- Local path expected by the loader: `data/enf/`
- Primary source for Leiva (the anchoring layer).
- Week 2: Leiva runs a quick statistical profile to find edge cases.

## Model-level energy benchmarks

AIEnergyScore, ML.ENERGY (arXiv 2310.03003), Watt Counts. Per-model energy
signatures, useful for Marchisano (attacks that mimic named models) and
Hendricks (workload-class calibration).

## Until the real data is here

`data_loaders.synthetic_power_profile` and `synthetic_enf` generate
plausible-shaped traces so you can develop on day one. They are shapes, not
measurements. Swap in real data before any number goes in the paper.
