"""I/O curve decider for NeuroSimo.

Determines the stimulation intensity (S50) to use in the later scripts.

  - Single pulses at 5%, 10%, ... 100% MSO (20 levels), 2 trials per level = 40 trials
  - Trial order is randomised (seeded per subject) and saved to CSV BEFORE the first trial
  - Every trial is predetermined (scheduled ITI); no phase/EEG-guided triggering
  - Each completed trial is appended to a trials CSV (MEP peak-to-peak + baseline RMS)
  - After the last trial, a constrained 4-parameter Boltzmann curve is fitted to the mean MEP
    per intensity (port of the MATLAB lsqcurvefit code, see io_curve_fit.py), and S50, the
    figure (png + pdf) and a result JSON are saved

Protocol: one stage with 40 trials (the decider raises an error if it is asked for more).

Outputs, in results/<subject_id>/io_curve/:
    io_curve_trial_order.csv        planned order (trial, intensity_pct, seed)
    io_curve_trials.csv             one row per delivered pulse
    sub-XXX_IO_S50_<value>.png/pdf  figure
    sub-XXX_IO_result.json          S50, MEP_min, MEP_max, slope k, QC counts

If a session is stopped early, or you want a different QC cutoff, refit offline:
    python io_curve_fit.py results/<id>/io_curve/io_curve_trials.csv --subject sub-XXX --cutoff 15
"""

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from io_curve_fit import analyse
from util.magventure_tms import MagVentureTMS
from util.magventure_tms_mock import MockTMS

# ---------------------------------------------------------------------------
# Protocol parameters
# ---------------------------------------------------------------------------
INTENSITIES = list(range(5, 101, 5))     # % MSO: 5, 10, ..., 100
TRIALS_PER_INTENSITY = 2
N_TRIALS = len(INTENSITIES) * TRIALS_PER_INTENSITY   # 40

# Inter-trial interval (s), uniform. Matches sc.isi = [2 3] in the MATLAB acquisition code.
ITI_RANGE = (2.0, 3.0)

SEED = 301

# MEP / QC windows relative to the pulse (s), on EMG channel 0.
# Baseline window and threshold from the MATLAB acquisition code. MEP window is 25-50 ms to match the POPSTAR deciders.
MEP_WINDOW = (0.025, 0.050)
BASELINE_WINDOW = (-0.100, -0.010)
# Trials whose baseline RMS (uV, mean-removed) exceeds this are excluded from the fit.
# The MATLAB code defines sc.baselineThreshold = 15 uV; check this is the cutoff your fitting step used.
BASELINE_RMS_CUTOFF = 15.0

BUFFER_TOLERANCE = 0.005


class Decider:
    def __init__(self, subject_id: int, num_eeg_channels: int, num_emg_channels: int, sampling_frequency: int,
                 runtime_params: dict[str, Any]):
        self.subject_id = subject_id
        self.num_eeg_channels = num_eeg_channels
        self.num_emg_channels = num_emg_channels
        self.sampling_frequency = sampling_frequency
        self.subject_label = f"sub-{subject_id:03d}"
        self.n_trials = N_TRIALS

        self.mock_tms_device = runtime_params.get("mock_tms_device")
        self.tms = MagVentureTMS() if not self.mock_tms_device else MockTMS()

        self.iti_rng = np.random.default_rng((SEED, subject_id, 1))

        self.results_dir = Path("results") / str(subject_id) / "io_curve"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[IO] Saving to: {self.results_dir.resolve()}", flush=True)

        self.order_csv = self.results_dir / "io_curve_trial_order.csv"
        self.trials_csv = self.results_dir / "io_curve_trials.csv"
        self.trial_order = self._load_or_create_order()

        # Keep any earlier trials file that has data in it instead of overwriting it
        if self.trials_csv.exists() and sum(1 for _ in open(self.trials_csv)) > 1:
            backup = self.trials_csv.with_name(
                f"io_curve_trials_prev_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            os.replace(self.trials_csv, backup)
            print(f"[IO] Existing trials file kept as {backup}")

        self.csv_fields = [
            "trial", "stage", "intensity_pct", "iti", "trial_start_time", "target_time", "pulse_time",
            "mep_ptp_uv", "baseline_rms_uv", "rejected_baseline",
        ]
        with open(self.trials_csv, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields).writeheader()

        self.current_trial: dict = {}
        print(f"[IO] I/O curve decider ready  subject={subject_id}  trials={N_TRIALS} "
              f"({len(INTENSITIES)} intensities x {TRIALS_PER_INTENSITY})", flush=True)

    # ==================================================================
    # Configuration
    # ==================================================================
    def get_configuration(self) -> dict[str, Any]:
        return {
            "periodic_processing_interval": 0.05,
            "sample_window": [-0.1, 0.0],
            # must cover the baseline window and the MEP window
            "pulse_sample_window": [BASELINE_WINDOW[0] - BUFFER_TOLERANCE, MEP_WINDOW[1] + BUFFER_TOLERANCE],
            "warm_up_rounds": 0,
        }

    # ==================================================================
    # Trial order
    # ==================================================================
    def _load_or_create_order(self) -> list[int]:
        if self.order_csv.exists():
            with open(self.order_csv, "r", newline="") as f:
                order = [int(r["intensity_pct"]) for r in csv.DictReader(f)]
            expected = sorted(INTENSITIES * TRIALS_PER_INTENSITY)
            if sorted(order) != expected:
                raise ValueError(f"{self.order_csv} does not contain {TRIALS_PER_INTENSITY} trials at each of "
                                 f"{INTENSITIES[0]}-{INTENSITIES[-1]}%. Delete it to regenerate.")
            print(f"[IO] Reloaded existing trial order from {self.order_csv.resolve()}")
            return order

        seed = int(SEED + self.subject_id)
        rng = np.random.default_rng(seed)
        order = np.array(INTENSITIES * TRIALS_PER_INTENSITY)
        rng.shuffle(order)
        order = [int(v) for v in order]
        with open(self.order_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "intensity_pct", "seed"])
            for i, v in enumerate(order, 1):
                writer.writerow([i, v, seed])
        print(f"[IO] Generated randomised trial order (seed {seed}) -> {self.order_csv.resolve()}")
        return order

    # ==================================================================
    # Trial preparation: set the intensity and schedule the pulse
    # ==================================================================
    def prepare_trial(self, start_time: float, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        if trial_in_stage >= self.n_trials:
            raise ValueError(
                f"Trial {trial_in_stage + 1} requested, but the I/O curve has {self.n_trials} planned trials. "
                f"The protocol should have a single stage of {self.n_trials} trials."
            )

        intensity = int(self.trial_order[trial_in_stage])
        iti = float(self.iti_rng.uniform(*ITI_RANGE))

        self.tms.set_single_pulse(intensity)

        self.current_trial = {
            "trial": trial_in_stage + 1,
            "stage": stage_name,
            "intensity_pct": intensity,
            "iti": iti,
            "trial_start_time": start_time,
            "target_time": start_time + iti,
        }
        print(f"[IO] Trial {trial_in_stage + 1}/{self.n_trials}: {intensity}% MSO, pulse in {iti:.2f}s", flush=True)
        return {"trigger_offset": iti}

    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int,
            is_warm_up: bool) -> dict[str, Any] | None:
        # All pulses are scheduled in prepare_trial; nothing to decide online.
        return None

    # ==================================================================
    # Pulse processing: MEP + baseline QC
    # ==================================================================
    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        mep = np.nan
        mep_mask = (time_offsets >= MEP_WINDOW[0]) & (time_offsets <= MEP_WINDOW[1])
        if mep_mask.sum() >= 2:
            mep = float(np.ptp(emg_buffer[mep_mask, 0]))

        baseline_rms = np.nan
        base_mask = (time_offsets >= BASELINE_WINDOW[0]) & (time_offsets <= BASELINE_WINDOW[1])
        if base_mask.sum() >= 2:
            seg = emg_buffer[base_mask, 0]
            baseline_rms = float(np.sqrt(np.mean((seg - seg.mean()) ** 2)))   # mean removed: DC offset isn't noise

        if not self.current_trial or self.current_trial.get("trial") != trial_in_stage + 1:
            # Pulse for a trial we didn't prepare (shouldn't happen); still log it with the planned intensity
            self.current_trial = {"trial": trial_in_stage + 1, "stage": stage_name,
                                  "intensity_pct": self.trial_order[trial_in_stage]}

        rejected = bool(np.isnan(mep) or np.isnan(baseline_rms) or baseline_rms > BASELINE_RMS_CUTOFF)
        self.current_trial.update({
            "pulse_time": reference_time,
            "mep_ptp_uv": mep,
            "baseline_rms_uv": baseline_rms,
            "rejected_baseline": rejected,
        })
        with open(self.trials_csv, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.csv_fields, extrasaction="ignore").writerow(self.current_trial)

        print(f"[IO] Trial {trial_in_stage + 1}/{self.n_trials} done: {self.current_trial['intensity_pct']}% | "
              f"MEP={mep:.1f} uV | baseline RMS={baseline_rms:.1f} uV{'  (REJECTED)' if rejected else ''}", flush=True)

        if trial_in_stage + 1 >= self.n_trials:
            self.run_fit()

    # ==================================================================
    # Fit
    # ==================================================================
    def run_fit(self) -> None:
        print("[IO] All trials complete - fitting Boltzmann curve...", flush=True)
        try:
            analyse(str(self.trials_csv), str(self.results_dir), self.subject_label,
                    cutoff=BASELINE_RMS_CUTOFF, mep_window=MEP_WINDOW, baseline_window=BASELINE_WINDOW)
        except Exception as exc:   # never let analysis problems break the session; data are already saved
            print(f"[IO] Fit/plot failed: {exc!r}. Trials are saved; refit with io_curve_fit.py.", flush=True)
