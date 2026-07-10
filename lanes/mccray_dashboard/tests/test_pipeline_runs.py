"""
test_pipeline_runs.py
---------------------
One test per valid pipeline run -- every dataset / node-count / SLURM-id
combination that scripts/run_microverse.py can actually ingest (46 total,
enumerated 2026-07 from the real folders under 00_raw_datasets/).

Each test performs the same discovery stage 1 of the pipeline relies on:
discover_slurm_ids() must surface the run's SLURM id, and
discover_nlr_pairs() must return exactly one NVML+RAPL pair per node.
A failure here means that run would break run_microverse.py before the
dashboard ever sees data.

The datasets root defaults to ~/Downloads/00_raw_datasets and can be
overridden with the MICROVERSE_RAW_DATASETS env var. All tests skip if
the root isn't present (e.g. on CI, which doesn't carry the raw data).
"""

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from microverse_core.data_loaders import (
    build_combined_records,
    combined_smooth,
    discover_nlr_pairs,
    load_enf,
    load_nlr_multi,
)
from run_microverse import discover_slurm_ids

DATASETS_ROOT = Path(
    os.environ.get(
        "MICROVERSE_RAW_DATASETS",
        str(Path.home() / "Downloads" / "00_raw_datasets"),
    )
)
ENF_FOLDER = Path(
    os.environ.get(
        "MICROVERSE_ENF_FOLDER",
        str(Path.home() / "Downloads" / "enf"),
    )
)

LLAMA2 = "training_llama2_70b_lora"
SDIFF = "training_stable_diffusion"


@pytest.mark.skipif(
    not DATASETS_ROOT.exists(),
    reason=f"raw datasets root not found: {DATASETS_ROOT} "
           f"(set MICROVERSE_RAW_DATASETS to override)",
)
class TestEveryPipelineRun:
    """Hits all 46 valid dataset/node/SLURM runs the pipeline can ingest."""

    def _check_run(self, dataset: str, node_folder: str, slurm_id: str):
        """Shared assertion body: the run's SLURM id is discoverable in its
        node folder and NLR pair discovery yields exactly one pair per node,
        exactly as stage_1_ingest_and_smooth() would consume it."""
        folder = DATASETS_ROOT / dataset / node_folder
        assert folder.is_dir(), f"node folder missing: {folder}"

        found_ids = discover_slurm_ids(folder)
        assert slurm_id in found_ids, (
            f"SLURM {slurm_id} not discoverable in {folder} "
            f"(found: {found_ids})"
        )

        pairs = discover_nlr_pairs(str(folder), slurm_id=slurm_id)
        expected_nodes = int("".join(c for c in node_folder if c.isdigit()))
        assert len(pairs) == expected_nodes, (
            f"{dataset}/{node_folder} slurm {slurm_id}: expected "
            f"{expected_nodes} NVML+RAPL pairs, got {len(pairs)}"
        )

    # ------------------------------------------------------------------
    # training_llama2_70b_lora / 2node (5 runs)
    # ------------------------------------------------------------------

    def test_llama2_2node_slurm_10742795(self):
        """Verifies the llama2_70b_lora 2-node SLURM 10742795 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "2node", "10742795")

    def test_llama2_2node_slurm_10742796(self):
        """Verifies the llama2_70b_lora 2-node SLURM 10742796 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "2node", "10742796")

    def test_llama2_2node_slurm_10742797(self):
        """Verifies the llama2_70b_lora 2-node SLURM 10742797 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "2node", "10742797")

    def test_llama2_2node_slurm_10742798(self):
        """Verifies the llama2_70b_lora 2-node SLURM 10742798 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "2node", "10742798")

    def test_llama2_2node_slurm_10742800(self):
        """Verifies the llama2_70b_lora 2-node SLURM 10742800 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "2node", "10742800")

    # ------------------------------------------------------------------
    # training_llama2_70b_lora / 4node (5 runs)
    # ------------------------------------------------------------------

    def test_llama2_4node_slurm_10742829(self):
        """Verifies the llama2_70b_lora 4-node SLURM 10742829 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "4node", "10742829")

    def test_llama2_4node_slurm_10742831(self):
        """Verifies the llama2_70b_lora 4-node SLURM 10742831 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "4node", "10742831")

    def test_llama2_4node_slurm_10742832(self):
        """Verifies the llama2_70b_lora 4-node SLURM 10742832 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "4node", "10742832")

    def test_llama2_4node_slurm_10742833(self):
        """Verifies the llama2_70b_lora 4-node SLURM 10742833 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "4node", "10742833")

    def test_llama2_4node_slurm_10742834(self):
        """Verifies the llama2_70b_lora 4-node SLURM 10742834 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "4node", "10742834")

    # ------------------------------------------------------------------
    # training_llama2_70b_lora / 8node (6 runs)
    # ------------------------------------------------------------------

    def test_llama2_8node_slurm_10742766(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742766 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742766")

    def test_llama2_8node_slurm_10742817(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742817 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742817")

    def test_llama2_8node_slurm_10742818(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742818 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742818")

    def test_llama2_8node_slurm_10742819(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742819 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742819")

    def test_llama2_8node_slurm_10742820(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742820 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742820")

    def test_llama2_8node_slurm_10742821(self):
        """Verifies the llama2_70b_lora 8-node SLURM 10742821 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "8node", "10742821")

    # ------------------------------------------------------------------
    # training_llama2_70b_lora / 16node (5 runs)
    # ------------------------------------------------------------------

    def test_llama2_16node_slurm_10742842(self):
        """Verifies the llama2_70b_lora 16-node SLURM 10742842 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "16node", "10742842")

    def test_llama2_16node_slurm_10742843(self):
        """Verifies the llama2_70b_lora 16-node SLURM 10742843 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "16node", "10742843")

    def test_llama2_16node_slurm_10742844(self):
        """Verifies the llama2_70b_lora 16-node SLURM 10742844 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "16node", "10742844")

    def test_llama2_16node_slurm_10742845(self):
        """Verifies the llama2_70b_lora 16-node SLURM 10742845 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "16node", "10742845")

    def test_llama2_16node_slurm_10742846(self):
        """Verifies the llama2_70b_lora 16-node SLURM 10742846 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(LLAMA2, "16node", "10742846")

    # ------------------------------------------------------------------
    # training_stable_diffusion / 1node (5 runs)
    # ------------------------------------------------------------------

    def test_sdiff_1node_slurm_10742933(self):
        """Verifies the stable_diffusion 1-node SLURM 10742933 run is ingestible with exactly 1 NVML+RAPL node pair."""
        self._check_run(SDIFF, "1node", "10742933")

    def test_sdiff_1node_slurm_10742935(self):
        """Verifies the stable_diffusion 1-node SLURM 10742935 run is ingestible with exactly 1 NVML+RAPL node pair."""
        self._check_run(SDIFF, "1node", "10742935")

    def test_sdiff_1node_slurm_10742937(self):
        """Verifies the stable_diffusion 1-node SLURM 10742937 run is ingestible with exactly 1 NVML+RAPL node pair."""
        self._check_run(SDIFF, "1node", "10742937")

    def test_sdiff_1node_slurm_10742938(self):
        """Verifies the stable_diffusion 1-node SLURM 10742938 run is ingestible with exactly 1 NVML+RAPL node pair."""
        self._check_run(SDIFF, "1node", "10742938")

    def test_sdiff_1node_slurm_10742939(self):
        """Verifies the stable_diffusion 1-node SLURM 10742939 run is ingestible with exactly 1 NVML+RAPL node pair."""
        self._check_run(SDIFF, "1node", "10742939")

    # ------------------------------------------------------------------
    # training_stable_diffusion / 2node (6 runs)
    # ------------------------------------------------------------------

    def test_sdiff_2node_slurm_10742951(self):
        """Verifies the stable_diffusion 2-node SLURM 10742951 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742951")

    def test_sdiff_2node_slurm_10742971(self):
        """Verifies the stable_diffusion 2-node SLURM 10742971 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742971")

    def test_sdiff_2node_slurm_10742974(self):
        """Verifies the stable_diffusion 2-node SLURM 10742974 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742974")

    def test_sdiff_2node_slurm_10742976(self):
        """Verifies the stable_diffusion 2-node SLURM 10742976 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742976")

    def test_sdiff_2node_slurm_10742977(self):
        """Verifies the stable_diffusion 2-node SLURM 10742977 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742977")

    def test_sdiff_2node_slurm_10742978(self):
        """Verifies the stable_diffusion 2-node SLURM 10742978 run is ingestible with exactly 2 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "2node", "10742978")

    # ------------------------------------------------------------------
    # training_stable_diffusion / 4node (5 runs)
    # ------------------------------------------------------------------

    def test_sdiff_4node_slurm_10742981(self):
        """Verifies the stable_diffusion 4-node SLURM 10742981 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "4node", "10742981")

    def test_sdiff_4node_slurm_10742982(self):
        """Verifies the stable_diffusion 4-node SLURM 10742982 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "4node", "10742982")

    def test_sdiff_4node_slurm_10742983(self):
        """Verifies the stable_diffusion 4-node SLURM 10742983 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "4node", "10742983")

    def test_sdiff_4node_slurm_10742986(self):
        """Verifies the stable_diffusion 4-node SLURM 10742986 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "4node", "10742986")

    def test_sdiff_4node_slurm_10742988(self):
        """Verifies the stable_diffusion 4-node SLURM 10742988 run is ingestible with exactly 4 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "4node", "10742988")

    # ------------------------------------------------------------------
    # training_stable_diffusion / 8node (5 runs)
    # ------------------------------------------------------------------

    def test_sdiff_8node_slurm_10742992(self):
        """Verifies the stable_diffusion 8-node SLURM 10742992 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "8node", "10742992")

    def test_sdiff_8node_slurm_10742993(self):
        """Verifies the stable_diffusion 8-node SLURM 10742993 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "8node", "10742993")

    def test_sdiff_8node_slurm_10742994(self):
        """Verifies the stable_diffusion 8-node SLURM 10742994 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "8node", "10742994")

    def test_sdiff_8node_slurm_10742995(self):
        """Verifies the stable_diffusion 8-node SLURM 10742995 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "8node", "10742995")

    def test_sdiff_8node_slurm_10742996(self):
        """Verifies the stable_diffusion 8-node SLURM 10742996 run is ingestible with exactly 8 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "8node", "10742996")

    # ------------------------------------------------------------------
    # training_stable_diffusion / 16node (4 runs)
    # ------------------------------------------------------------------

    def test_sdiff_16node_slurm_10743000(self):
        """Verifies the stable_diffusion 16-node SLURM 10743000 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "16node", "10743000")

    def test_sdiff_16node_slurm_10743001(self):
        """Verifies the stable_diffusion 16-node SLURM 10743001 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "16node", "10743001")

    def test_sdiff_16node_slurm_10743003(self):
        """Verifies the stable_diffusion 16-node SLURM 10743003 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "16node", "10743003")

    def test_sdiff_16node_slurm_10743005(self):
        """Verifies the stable_diffusion 16-node SLURM 10743005 run is ingestible with exactly 16 NVML+RAPL node pairs."""
        self._check_run(SDIFF, "16node", "10743005")


# All 46 valid runs, as (dataset, node_folder, slurm_id) -- the same
# combinations TestEveryPipelineRun covers method-by-method above.
RUNS = [
    (LLAMA2, "2node", "10742795"),
    (LLAMA2, "2node", "10742796"),
    (LLAMA2, "2node", "10742797"),
    (LLAMA2, "2node", "10742798"),
    (LLAMA2, "2node", "10742800"),
    (LLAMA2, "4node", "10742829"),
    (LLAMA2, "4node", "10742831"),
    (LLAMA2, "4node", "10742832"),
    (LLAMA2, "4node", "10742833"),
    (LLAMA2, "4node", "10742834"),
    (LLAMA2, "8node", "10742766"),
    (LLAMA2, "8node", "10742817"),
    (LLAMA2, "8node", "10742818"),
    (LLAMA2, "8node", "10742819"),
    (LLAMA2, "8node", "10742820"),
    (LLAMA2, "8node", "10742821"),
    (LLAMA2, "16node", "10742842"),
    (LLAMA2, "16node", "10742843"),
    (LLAMA2, "16node", "10742844"),
    (LLAMA2, "16node", "10742845"),
    (LLAMA2, "16node", "10742846"),
    (SDIFF, "1node", "10742933"),
    (SDIFF, "1node", "10742935"),
    (SDIFF, "1node", "10742937"),
    (SDIFF, "1node", "10742938"),
    (SDIFF, "1node", "10742939"),
    (SDIFF, "2node", "10742951"),
    (SDIFF, "2node", "10742971"),
    (SDIFF, "2node", "10742974"),
    (SDIFF, "2node", "10742976"),
    (SDIFF, "2node", "10742977"),
    (SDIFF, "2node", "10742978"),
    (SDIFF, "4node", "10742981"),
    (SDIFF, "4node", "10742982"),
    (SDIFF, "4node", "10742983"),
    (SDIFF, "4node", "10742986"),
    (SDIFF, "4node", "10742988"),
    (SDIFF, "8node", "10742992"),
    (SDIFF, "8node", "10742993"),
    (SDIFF, "8node", "10742994"),
    (SDIFF, "8node", "10742995"),
    (SDIFF, "8node", "10742996"),
    (SDIFF, "16node", "10743000"),
    (SDIFF, "16node", "10743001"),
    (SDIFF, "16node", "10743003"),
    (SDIFF, "16node", "10743005"),
]


def _run_label(run: tuple) -> str:
    dataset, node_folder, slurm_id = run
    short = "llama2" if dataset == LLAMA2 else "sdiff"
    return f"{short}-{node_folder}-{slurm_id}"


@pytest.fixture(scope="module")
def smoothed_enf():
    """One real ENF hour, loaded and smoothed once for the whole module --
    which ENF file doesn't vary per run, so any present one will do."""
    if not ENF_FOLDER.exists():
        pytest.skip(f"ENF folder not found: {ENF_FOLDER} "
                    f"(set MICROVERSE_ENF_FOLDER to override)")
    enf_files = sorted(ENF_FOLDER.glob("Dev*_ENF_Hr*.csv"))
    if not enf_files:
        pytest.skip(f"no Dev*_ENF_Hr*.csv files in {ENF_FOLDER}")
    return combined_smooth(load_enf(str(enf_files[0])))


@pytest.mark.skipif(
    not DATASETS_ROOT.exists(),
    reason=f"raw datasets root not found: {DATASETS_ROOT} "
           f"(set MICROVERSE_RAW_DATASETS to override)",
)
class TestEveryPipelineRunFullIngest:
    """Slow integration pass: actually runs stage 1 (parse every NVML+RAPL
    log, merge with smoothed ENF) for all 46 runs -- the same work
    stage_1_ingest_and_smooth() does, minus writing the JSONL to disk.
    Catches malformed/truncated logs that pure discovery can't see."""

    @pytest.mark.parametrize("dataset,node_folder,slurm_id", RUNS,
                             ids=[_run_label(r) for r in RUNS])
    def test_stage_1_full_ingest(self, smoothed_enf, dataset, node_folder,
                                 slurm_id):
        """Fully ingests this run exactly as pipeline stage 1 would --
        parsing every one of its NVML+RAPL logs and merging them with the
        smoothed ENF -- and asserts one complete record per ENF sample with
        FRQ plus at least one metric column for every node in the run."""
        folder = DATASETS_ROOT / dataset / node_folder
        pairs = discover_nlr_pairs(str(folder), slurm_id=slurm_id)
        assert pairs, f"no NLR pairs for {dataset}/{node_folder} {slurm_id}"

        node_windows = load_nlr_multi(pairs)
        records = build_combined_records(smoothed_enf, node_windows)

        assert len(records) == len(smoothed_enf), (
            f"expected {len(smoothed_enf)} records (one per ENF sample), "
            f"got {len(records)}"
        )
        first = records[0]
        assert "index" in first and "FRQ" in first
        for node_id, _nvml, _rapl in pairs:
            assert any(k.startswith(f"{node_id}_") for k in first), (
                f"no metric columns for node {node_id} in combined record"
            )
