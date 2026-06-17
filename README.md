# Secure AI Data Center Microverse (AFRL Summer 2026)

Anchored digital twins for trusted operations. We anchor a data-center digital
twin to authenticated physical reality (ENF and power signatures) so its state
is verifiable rather than just plausible, then we attack it and measure how
well the verification holds up.

This repository is the shared codebase for all five lanes. `microverse_core` is
the integration layer (Lance's lane): the contracts every component builds
against, the Blender bridge, a placeholder twin, data loaders, a file bus, and
the scoring metrics.

## The data flow

```
  ENF dataset ------------+
                          v
  NLR power profiles --> Hendricks DT --> Leiva verification --> McCray dashboard
                          ^                     ^
                          |                     |
              Amzad visual realism     Marchisano attacks (replay/injection/drift)
                                                 |
                                                 +--> metrics (precision/recall/F1)
```

Everything crosses a lane boundary as a record defined in
`microverse_core/contracts.py`. Read that file first. It is the single most
important thing to agree on, because four people break if it changes silently.

## Quickstart

```bash
# 1. install for development (pure stdlib, but this makes imports clean)
pip install -e ".[dev]"

# 2. run the whole pipeline with no Blender, no real data
python scripts/smoke_test.py        # should print OK

# 3. run the tests
pytest

# 4. build the placeholder twin (needs Blender on your PATH)
blender --background --python scripts/build_starter_scene.py -- --save twin.blend
```

If `smoke_test.py` prints OK, the contracts, file bus, and metrics are wired
correctly and you have a working mental model of the system before you have
written a line of your own lane's code.

## Layout

```
microverse_core/        the integration layer (Lance owns this)
  contracts.py          record shapes shared by every lane  <-- read first
  blender_bridge.py     the only file that touches bpy
  scene_builder.py      PLACEHOLDER twin, replaced by Hendricks in week 2
  data_loaders.py       NLR + ENF loaders, with synthetic fallback
  io_records.py         JSONL file bus under runs/
  metrics.py            shared precision/recall/F1/time-to-detection
scripts/
  build_starter_scene.py  run inside Blender
  smoke_test.py           end-to-end demo, no Blender
lanes/                  one folder + README per intern, your code goes here
tests/
data/                   datasets live here locally, never committed
```

## How we work (from the project guide)

- One repo. Push to your own branch, merge through a pull request after a quick
  review. No code lives only on a laptop.
- Every module gets a README with three short sections: what it does, how to
  run it, who to ask.
- Commit messages are specific. Not "fixed bug" but "fixed off-by-one in ENF
  window alignment that caused replay false negatives."
- Friday status note: 3 to 5 lines on what you did, what you are blocked on,
  what is next.
- Stuck for more than 4 hours on the same thing, stop and ask.

## Who to ask

| Topic | Person |
|---|---|
| Repo setup, integration, Blender mechanics, where my code goes | Lance |
| DT physical model, how NLR profiles map to the twin | Dr. Xiang, Hendricks |
| Anchoring / verification methodology, Microverse architecture | Dr. Qu |
| Adversarial evaluation design, attack methodology | Dr. Xiang |
| Anything you do not know who else to ask | Dr. Chen |
| Admin / program | Dr. Ardiles-Cruz |

## Branch naming

`lane/<name>/<short-topic>`, for example `lane/leiva/anchor-extractor` or
`lane/integration/file-bus`. Keep one topic per branch so reviews stay small.
