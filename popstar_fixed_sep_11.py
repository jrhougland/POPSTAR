"""
RL Phastimate Decider for NeuroSimo
=====================================
Combines Phastimate (real-time EEG phase estimation) with a DQN+LSTM agent
that learns which of 8 discrete EEG phase targets maximises MEP amplitude.

Session structure (mirrors POPSTAR MATLAB):
  Block 0 — RL training   (N_RL_TRIALS trials, agent trains online)
  Block 1 — Trough test   (fixed phase = π)
  Block 2 — Random test   (random phase from CSV)

References:
  Zrenner et al. (2020) NeuroImage 214, 116761  — Phastimate algorithm
  POPSTAR 2025 MATLAB codebase — RL environment & reward design
"""

import csv
import os
import time
import socket
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
from scipy.signal import filtfilt, hilbert
from spectrum import aryule

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# --- EEG spatial filter (C3 Hjorth) ---
C3_CHANNEL_INDEX = 4
REFERENCE_CHANNEL_INDICES = [20, 22, 24, 26]
REFERENCE_WEIGHT = 0.25

# --- Phastimate parameters ---
DEFAULT_HILBERT_WINDOW_SIZE = 128
DEFAULT_EDGE_SAMPLES = 64
DEFAULT_AR_MODEL_ORDER = 25
DEFAULT_DOWNSAMPLE_RATIO = 10

# --- Timing ---
DEFAULT_PROCESSING_INTERVAL_SECONDS = 0.05
DEFAULT_BUFFER_SIZE_SECONDS = 0.5
MINIMUM_TRIGGER_DELAY_SECONDS = 0.005
FIRST_ROUND_WAIT_SECONDS = 0.5

# --- Phase targeting ---
DEFAULT_PHASE_TOLERANCE = np.pi / 40

# --- RL action space (from POPSTAR logic) ---
N_ACTIONS = 8
ACTION_TO_PHASE = {k: (k - 5) * 0.25 * np.pi for k in range(1, N_ACTIONS + 1)}

# --- Session structure ---
# The protocol (popstar.yaml) defines how many trials each stage actually has; these
# values are used for the RL 'done' flag and for printouts, so they should match it.
N_RL_TRIALS = 800
N_TEST_TRIALS = 200

# --- DQN hyper-parameters (mirrors MATLAB agentOptions) ---
REPLAY_BUFFER_SIZE = 100    #was 50
MINI_BATCH_SIZE = 32    #was 15
LEARN_RATE = 1e-3
GRAD_CLIP = 1.0
DISCOUNT_FACTOR = 1.0
TARGET_SMOOTH = 5e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.002 #800 trials total - exploration about halfway
SEQUENCE_LENGTH = 2
REWARD_CAP = 10000.0

# Load MATLAB filter coefficients
mat_data = loadmat('data/filter_coeffs.mat')
BANDPASS_FILTER_COEFFICIENTS = np.array(mat_data['coeffs'].flatten())

RANDOM_PHASES_PATH = "data/random_phases.csv"

# save training here??
SAVE_PATH = "data"


# ---------------------------------------------------------------------------
# NEURAL NETWORK (CRITIC)
# ---------------------------------------------------------------------------
class DQNLSTMNet(nn.Module):
    """
    Q-network replicating MATLAB criticNW:
      sequenceInput(2) → FC(12, LeakyReLU0.3) → FC(8, LeakyReLU0.3)
      → LSTM(10) → FC(8, LeakyReLU0.3) → FC(8) [Q-values]
    """
    # define layers of NN
    def __init__(self, input_dim: int = 2, hidden_dim: int = 10, n_actions: int = N_ACTIONS):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, 12)
        self.fc2  = nn.Linear(12, 8)
        self.lstm = nn.LSTM(8, hidden_dim, batch_first=True) # memory layer
        self.fc3  = nn.Linear(hidden_dim, 8)
        self.out  = nn.Linear(8, n_actions) # Q-values (actions from 1 to 8)

    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flow of data through NN layers
        Args:
            x : (batch, seq_len, 2)
            hidden : optional LSTM hidden state
        Returns:
            q_values : (batch, n_actions)
            hidden : updated LSTM hidden state
        """
        x = F.leaky_relu(self.fc1(x), negative_slope=0.3)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.3)
        lstm_out, hidden = self.lstm(x, hidden)
        x = F.leaky_relu(self.fc3(lstm_out[:, -1, :]), negative_slope=0.3)
        return self.out(x), hidden


# ---------------------------------------------------------------------------
# REPLAY BUFFER (MEMORY)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, seq_len: int):
        self.buffer  = deque(maxlen=capacity) # deque (list with max size of 50)
        self.seq_len = seq_len

    def push(self, obs_seq: np.ndarray, action: int, reward: float, next_obs_seq: np.ndarray, done: bool) -> None:
        self.buffer.append((obs_seq, action, reward, next_obs_seq, done))

    # look at random events to learn
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(acts),
            torch.FloatTensor(rews),
            torch.FloatTensor(np.array(next_obs)),
            torch.FloatTensor(dones),
        )

    # stored memories
    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN AGENT
# online net (learns) vs target net (stable)
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Online DQN agent with soft target-network updates.
    ε-greedy with linear decay per trial, buffer persists across episodes.
    """
    def __init__(self, device: torch.device):
        self.device  = device
        self.epsilon = EPSILON_START
        self.seq_len = SEQUENCE_LENGTH

        self.online_net = DQNLSTMNet().to(device)
        self.target_net = DQNLSTMNet().to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARN_RATE)
        self.replay    = ReplayBuffer(REPLAY_BUFFER_SIZE, self.seq_len)

        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._obs_window: deque = deque(
            [np.zeros(2)] * self.seq_len, maxlen=self.seq_len
        )
    
    # clear LSTM (short term) memory
    def reset_episode(self) -> None:
        self._hidden = None
        self._obs_window = deque(
            [np.zeros(2)] * self.seq_len, maxlen=self.seq_len
        )

    def select_action(self, obs: np.ndarray) -> int:
        """
        ε-greedy selection. Returns action from 1 to 8.
        Updates the internal observation window with obs before deciding.
        """
        self._obs_window.append(obs)
        if np.random.rand() < self.epsilon: # exploration
            return np.random.randint(1, N_ACTIONS + 1)

        #exploitation (highest Q-value)
        seq = torch.FloatTensor(np.array(self._obs_window)).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            q_vals, self._hidden = self.online_net(seq, self._hidden)
        self.online_net.train()
        return int(q_vals.argmax(dim=1).item()) + 1  # back to 1-indexed

    def store_and_train(self, obs_seq: np.ndarray, action: int, reward: float, next_obs_seq: np.ndarray, done: bool) -> Optional[float]:
        """
        Push experience (using pre-built sequences) and run one gradient step
        if the buffer has enough samples.  Returns loss or None.
        """
        self.replay.push(obs_seq, action - 1, reward, next_obs_seq, done) #save to buffer
        if len(self.replay) < MINI_BATCH_SIZE:
            print(f"Not enough samples in replay buffer, skipping training")
            return None
        return self._train_step()

    # gradually lower epsilon (no random picking)
    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_MIN, self.epsilon - EPSILON_DECAY)

    def save(self, path: str) -> None:
        torch.save(
            {
                "online_net": self.online_net.state_dict(), # NN weights
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(), # training
                #"epsilon": self.epsilon, #remove so we don't store
            },
            path,
        )
        print(f"[DQNAgent] Saved to {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        #self.epsilon = ckpt["epsilon"]
        print(f"[DQNAgent] Loaded from {path}")

    def _train_step(self) -> float:
        # sample a batch of past experiences
        obs_b, act_b, rew_b, next_obs_b, done_b = self.replay.sample(MINI_BATCH_SIZE)
        obs_b = obs_b.to(self.device)
        act_b = act_b.to(self.device)
        rew_b = rew_b.to(self.device)
        next_obs_b = next_obs_b.to(self.device)
        done_b = done_b.to(self.device)

        # Current Q-values for taken actions
        q_vals, _ = self.online_net(obs_b)
        q_taken = q_vals.gather(1, act_b.unsqueeze(1)).squeeze(1)

        # Target Q-values 
        with torch.no_grad():
            q_next, _ = self.target_net(next_obs_b)
            # Q-learning
            q_target  = rew_b + DISCOUNT_FACTOR * q_next.max(dim=1).values * (1.0 - done_b)

        loss = F.mse_loss(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP)
        self.optimizer.step()

        # Soft target network update: Polyak averaging
        for p_online, p_target in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            p_target.data.copy_(
                TARGET_SMOOTH * p_online.data + (1.0 - TARGET_SMOOTH) * p_target.data
            )
        return loss.item()


# ---------------------------------------------------------------------------
# DECIDER  (NeuroSimo interface)
# ---------------------------------------------------------------------------
class Decider:
    """
    NeuroSimo Decider implementing RL-guided TMS phase targeting.

    Session blocks:
      0 — RL training  : agent selects phase, learns from MEP feedback
      1 — Trough test  : fixed phase = π
      2 — Random test  : random phase from CSV

    MEP feedback contract
    ---------------------
    After each TMS pulse, the external EMG pipeline must call:
        decider.receive_mep(amplitude_uV)
    """

    def __init__(self, subject_id: int, num_eeg_channels: int, num_emg_channels: int, sampling_frequency: float,
                 runtime_params):
        """
        Initialize the Decider with parameters and filter design.
        
        Args:
            subject_id: Subject ID (a string between 000 and 999)
            num_eeg_channels: Number of EEG channels (unused but kept for interface compatibility)
            num_emg_channels: Number of EMG channels (unused but kept for interface compatibility)
            sampling_frequency: Sampling frequency in Hz
        """
        self.sampling_frequency = sampling_frequency
        self.subject_id = "test_S" + f"{subject_id:03d}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Print if GPU is available
        if self.device.type == "cuda":
            print("GPU is available, using GPU")
        else:
            print("GPU is not available, using CPU")

        # Phastimate parameters 
        self.hilbert_window_size = DEFAULT_HILBERT_WINDOW_SIZE
        self.edge_samples = DEFAULT_EDGE_SAMPLES
        self.ar_model_order = DEFAULT_AR_MODEL_ORDER
        self.downsample_ratio = DEFAULT_DOWNSAMPLE_RATIO

        # Processing timing
        self.processing_interval_seconds = DEFAULT_PROCESSING_INTERVAL_SECONDS
        self.buffer_size_seconds = DEFAULT_BUFFER_SIZE_SECONDS
        self.buffer_size_samples = int(self.buffer_size_seconds * sampling_frequency)
        print(f"Buffer size in samples: {self.buffer_size_samples}")

        # Filter
        self.bandpass_filter_coefficients = BANDPASS_FILTER_COEFFICIENTS

        # NEW!!!!!
        self._action_counts = np.zeros(N_ACTIONS + 1)  # index 1-8
        self._action_rewards = np.zeros(N_ACTIONS + 1)  # cumulative reward per action
        self._best_rl_phase = np.pi

        # Phase tolerance
        self.phase_tolerance    = DEFAULT_PHASE_TOLERANCE
        self.max_future_samples = int(self.edge_samples / 2)
        #self.last_phase_error   = None

        # Session blocks
        self.best_phase_test_first = runtime_params["best_phase_test_first"]
        self.trough_test = runtime_params["trough_test"]

        print("Best phase test first" if self.best_phase_test_first else "Trough/random test first")
        print("Using trough test" if self.trough_test else "Using random test")

        rl_block = {
            "name": "RL training",
            "n_trials": N_RL_TRIALS,
            "mode": "rl",
        }

        best_phase_block = {
            "name": "Best RL phase test",
            "n_trials": N_TEST_TRIALS,
            "mode": "best_rl",
        }

        other_test_block = {
            "name": "Trough test" if self.trough_test else "Random test",
            "n_trials": N_TEST_TRIALS,
            "mode": "trough" if self.trough_test else "random",
        }

        self._blocks = [rl_block]

        if self.best_phase_test_first:
            self._blocks.extend([best_phase_block, other_test_block])
        else:
            self._blocks.extend([other_test_block, best_phase_block])

        print(self._blocks)

        # Block state is bound to the protocol stages reported by the experiment
        # coordinator: the Nth stage of the protocol drives the Nth block here. The
        # coordinator is also the authority on the trial number within a stage, so the
        # decider never counts trials itself (a scheduled trigger that never produces a
        # pulse is a failed trial and is retried by the coordinator).
        self._block_idx = 0
        self._current_stage_name: Optional[str] = None
        self._stages_warned_about_trial_count: set = set()

        # RL components 
        self._agent = DQNAgent(self.device)
        self._mep_history: List[float] = []

        # Pending experience: set when a trigger fires, consumed when MEP arrives
        self._pending_obs:     Optional[np.ndarray] = None
        self._pending_action:  Optional[int]        = None
        self._pending_obs_seq: Optional[np.ndarray] = None   # window snapshot at trigger time

        self._last_mep = 0.0
        self._last_reward = 0.0

        # MEP queue: receive_mep() deposits here, process_periodic() drains it
        self._mep_queue: deque = deque()

        # RL episode tracking
        self._steps_per_episode = 20 
        self._episode_step = 0
        self._episode_num = 0
        self._episode_rewards:  List[float] = []
        self._episode_total= 0.0

        # Random phases
        self._random_phases = self._load_random_phases(RANDOM_PHASES_PATH)

        self.earliest_next_pulse_time = None

        # Logging
        self._log: List[Dict] = []

        # Load saved agent checkpoint if available
        agent_path = os.path.join(SAVE_PATH, f"{self.subject_id}_dqn_agent.pt")
        if os.path.isfile(agent_path):
            self._agent.load(agent_path)

        print(
            f"[Decider] Initialised | device={self.device} | "
            f"RL trials={N_RL_TRIALS} | Test trials/block={N_TEST_TRIALS} |"
            f"Subject ID={self.subject_id}"
        )

    def __del__(self):
        print("\n=== Session finished — saving logs ===")
        self._save_all()


    def get_configuration(self) -> Dict[str, Any]:
        """
        Return the configuration for the processing interval and sample window.
        
        Returns:
            Dictionary containing processing configuration parameters
        """
        return {
            # Data configuration
            'sample_window': [-self.buffer_size_seconds, 0],
            'warm_up_rounds': 0,  # Number of warm-up rounds to perform (0 to disable)

            # Events
            'pulse_sample_window': [0.025, 0.05], # Custom window for pulse events TODO: change??

            # Periodic processing
            'periodic_processing_interval': self.processing_interval_seconds,
        }
    
    def _load_random_phases(self, path: str) -> np.ndarray:
        try:
            with open(path, "r") as f:
                phases = [float(row[0]) for row in csv.reader(f) if row]
            if len(phases) >= N_TEST_TRIALS:
                print(f"[Decider] Loaded {len(phases)} random phases from {path}")
                return np.array(phases)
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[Decider] Could not read random phases CSV: {exc}")

        np.random.seed(42)
        phases = np.random.uniform(-np.pi, np.pi, N_TEST_TRIALS)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            for p in phases:
                writer.writerow([p])
        print(f"[Decider] Generated and saved {len(phases)} random phases to {path}")
        return phases

    def _save_all(self) -> None:
        """Save agent weights and trial log at end of each block / session."""
        os.makedirs(SAVE_PATH, exist_ok=True)

        # Agent checkpoint
        agent_path = os.path.join(SAVE_PATH, f"{self.subject_id}_dqn_agent.pt")
        self._agent.save(agent_path)

        # Trial log
        log_path   = os.path.join(SAVE_PATH, f"{self.subject_id}_trial_log.csv")
        fieldnames = [
            "block", "block_name", "mode", "trial",
            "total_trial", "target_phase", "trigger_time",
            "mep", "reward", "isi", "earliest_next_pulse_time",
        ]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._log)
        print(f"[Decider] Trial log saved ({len(self._log)} rows) → {log_path}")

        # Episode rewards summary (RL episode)
        if self._episode_rewards:
            ep_path = os.path.join(SAVE_PATH, f"{self.subject_id}_episode_rewards.csv")
            with open(ep_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                for i, r in enumerate(self._episode_rewards, 1):
                    writer.writerow([i, r])
            print(f"[Decider] Episode rewards saved → {ep_path}")

    # changed from PHASTIMATE (LAVA_NEUROSIMO)
    # adapted for RL with feedback loop from MEP
    def process_periodic(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray,
            is_coil_at_target: bool, stage_name: str, trial_in_stage: int, is_warm_up: bool) -> dict[str, Any] | None:
        """
        Called every `processing_interval_seconds` while a trial is active.
        Steps:
          1. Respect the decider's own minimum inter-pulse interval
          2. Sync block state with the stage reported by the coordinator
          3. Run Phastimate
          4. Select target phase (RL / best RL / trough / random)
          5. Find trigger timing
          6. If trigger found, log and store pending RL experience

        Note that a trigger returned here is only an *attempt*: if it never results in a
        processed pulse, the coordinator retries the same trial, and this method is called
        again with the same `trial_in_stage`. The log therefore contains one row per
        attempt; only the successful attempt of a trial gets an MEP.
        """
        if self.earliest_next_pulse_time is not None and reference_time < self.earliest_next_pulse_time:
            return None

        # 1. Sync with the coordinator's stage; each stage maps to one block.
        if stage_name != self._current_stage_name:
            if not self._start_block_for_stage(stage_name):
                return None

        current_block = self._blocks[self._block_idx]
        mode = current_block["mode"]

        if trial_in_stage >= current_block["n_trials"] and stage_name not in self._stages_warned_about_trial_count:
            self._stages_warned_about_trial_count.add(stage_name)
            print(
                f"[Decider] Warning: stage '{stage_name}' is at trial {trial_in_stage + 1}, "
                f"beyond the {current_block['n_trials']} trials configured for block "
                f"'{current_block['name']}'. The protocol defines the number of trials; "
                f"check that the decider constants match the protocol."
            )

        # 2. Run Phastimate
        estimated_phases = self._run_phastimate(eeg_buffer)
        if estimated_phases is None:
            return None

        # 3. Select target phase
        target_phase = self._select_target_phase(mode, estimated_phases, trial_in_stage)
        if target_phase is None:
            return None

        # 4. Find trigger timing
        trigger_result = self._find_optimal_trigger_timing(
            estimated_phases, reference_time, target_phase
        )
        if trigger_result is None:
            return None

        # 5. Log the attempt; the pending RL experience was stored by _select_target_phase.
        block_trial = trial_in_stage + 1
        total_trial = self._trials_before_current_block() + block_trial

        self._log.append({
            "block":        self._block_idx,
            "block_name":   current_block["name"],
            "mode":         mode,
            "trial":        block_trial,
            "total_trial":  total_trial,
            "target_phase": target_phase,
            "trigger_time": trigger_result["trigger_offset"] + reference_time,
            "mep":          None,   # filled in when MEP arrives
            "reward":       None,
            "isi":          None,
            "earliest_next_pulse_time": None,
        })

        print(
            f"[Decider] Block={self._block_idx}({mode}) stage='{stage_name}' "
            f"trial={block_trial}/{current_block['n_trials']} "
            f"(global {total_trial}) | "
            f"target={np.degrees(target_phase):.1f}° | "
            f"trigger in {trigger_result['trigger_offset']:.3f}s",
            flush=True,
        )
        return trigger_result

    # ------------------------------------------------------------------ #
    #  Phastimate pipeline (unchanged from base code) 
    # ------------------------------------------------------------------ #

    def _run_phastimate(self, eeg_buffer: np.ndarray) -> Optional[np.ndarray]:
        """Spatial filter → preprocess → Phastimate."""
        c3 = self._extract_c3_referenced_data(eeg_buffer)
        if c3 is None:
            return None
        preprocessed    = self._preprocess_eeg_data(c3)
        phases, _       = self.phastimate(
            preprocessed,
            self.bandpass_filter_coefficients,
            [1.0],
            self.edge_samples,
            self.ar_model_order,
            self.hilbert_window_size,
        )
        return phases

    def _extract_c3_referenced_data(self, eeg_buffer: np.ndarray) -> Optional[np.ndarray]:
        try:
            c3_data        = eeg_buffer[:, C3_CHANNEL_INDEX]
            reference_data = np.sum(eeg_buffer[:, REFERENCE_CHANNEL_INDICES], axis=1)
            return c3_data - REFERENCE_WEIGHT * reference_data
        except IndexError:
            print("Error: EEG buffer does not have expected number of channels")
            return None

    def _preprocess_eeg_data(self, data: np.ndarray) -> np.ndarray:
        # Remove DC component
        demeaned = data - np.mean(data)

        # Downsample the data
        return demeaned[::self.downsample_ratio]

    def phastimate(self, data: np.ndarray, filter_b: np.ndarray, filter_a: List[float], 
                   edge_samples: int, ar_order: int, hilbert_window_size: int,
                   offset_correction: int = 0, iterations: Optional[int] = None, 
                   ar_method: str = 'aryule') -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate the phase of the EEG signal using autoregressive modeling and Hilbert transform.
        
        This is the core Phastimate algorithm that performs:
        1. Bandpass filtering of the input signal
        2. Autoregressive (AR) modeling for forward prediction
        3. Hilbert transform for phase extraction
        
        Args:
            data: Input EEG signal
            filter_b: Numerator coefficients of the bandpass filter
            filter_a: Denominator coefficients of the bandpass filter  
            edge_samples: Number of edge samples to remove after filtering
            ar_order: Order of the autoregressive model
            hilbert_window_size: Size of the window for Hilbert transform
            offset_correction: Offset correction (unused)
            iterations: Number of forward prediction iterations
            ar_method: Method for AR parameter estimation ('aryule')
            
        Returns:
            Tuple of (estimated_phases, estimated_amplitudes) or (None, None) if estimation fails
        """
        if iterations is None:
            iterations = edge_samples + int(np.ceil(hilbert_window_size / 2))

        min_padding_length = 3 * (max(len(filter_a), len(filter_b)) - 1)
        if data.shape[0] <= min_padding_length:
            print(f"Insufficient data for filtering: {data.shape[0]} <= {min_padding_length}")
            return None, None

        filtered_data = filtfilt(filter_b, filter_a, data)

        if filtered_data.shape[0] <= 2 * edge_samples:
            print(f"Insufficient data after edge removal: {filtered_data.shape[0]} <= {2 * edge_samples}")
            return None, None

        edge_removed_data = filtered_data[edge_samples:-edge_samples]

        ar_coefficients = self._fit_ar_model(edge_removed_data, ar_order, ar_method)
        if ar_coefficients is None:
            return None, None

        predicted_data = self._forward_predict(edge_removed_data, ar_coefficients, iterations)
        return self._extract_phase_amplitude(predicted_data, hilbert_window_size)

    def _fit_ar_model(self, data: np.ndarray, ar_order: int, ar_method: str) -> Optional[np.ndarray]:
        if len(data) < ar_order:
            print(f"Insufficient data for AR model: {len(data)} < {ar_order}")
            return None
        if ar_method == 'aryule':
            try:
                ar_params, _, _ = aryule(data, ar_order)
                return -1 * ar_params[::-1]
            except Exception as e:
                print(f"AR model fitting failed: {e}")
                return None
        else:
            raise ValueError(f'Unknown AR method: {ar_method}')

    def _forward_predict(self, data: np.ndarray, ar_coefficients: np.ndarray, iterations: int) -> np.ndarray:
        total_length = len(data) + iterations
        predicted_data = np.zeros(total_length)
        predicted_data[:len(data)] = data
        ar_order = len(ar_coefficients)
        for i in range(iterations):
            idx = len(data) + i
            predicted_data[idx] = np.sum(ar_coefficients * predicted_data[idx - ar_order:idx])
        return predicted_data

    def _extract_phase_amplitude(self, data: np.ndarray, window_size: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if data.shape[0] < window_size:
            print(f'Insufficient data for Hilbert transform: {data.shape[0]} < {window_size}')
            return None, None
        analysis_window = data[-window_size:]
        analytic_signal = hilbert(analysis_window)
        return np.angle(analytic_signal), np.abs(analytic_signal)

    # ------------------------------------------------------------------ #
    #  Phase selection                                           #
    # ------------------------------------------------------------------ #

    def _select_target_phase(self, mode: str, estimated_phases: np.ndarray, trial_in_stage: int) -> Optional[float]:
        """
        Return the target phase (radians) for this trial.

        Trough : always π
        Random : the value the CSV assigns to this trial (indexed by trial number, so a
                 retried trial repeats the same planned phase)
        RL     : agent selects one of 8 discrete phases based on the
                 current EEG phase and the most recent MEP amplitude.
                 The (obs, action, obs_window_snapshot) triple is stored
                 as pending so the reward can be assigned after the MEP
                 arrives via process_pulse().
        """
        if mode == "trough":
            return np.pi
        
        if mode == "best_rl":
            return self._best_rl_phase

        if mode == "random":
            return float(self._random_phases[trial_in_stage % len(self._random_phases)])

        # --- RL mode ---
        # Derive current instantaneous phase from the last Hilbert window
        analytic      = hilbert(estimated_phases[-self.hilbert_window_size:])
        current_phase = float(np.angle(analytic[-1]))

        obs    = np.array([current_phase, self._last_mep])
        action = self._agent.select_action(obs)   # also pushes obs into agent's window

        # Snapshot the observation window *after* select_action has updated it
        self._pending_obs     = obs.copy()
        self._pending_action  = action
        self._pending_obs_seq = np.array(list(self._agent._obs_window))  # (seq_len, 2)

        target_phase = ACTION_TO_PHASE[action]
        print(
            f"[RL] action={action} → target={np.degrees(target_phase):.1f}° | "
            f"ε={self._agent.epsilon:.3f}"
        )
        return target_phase

    # ------------------------------------------------------------------ #
    #  Internal: trigger timing (unchanged from base code)                #
    # ------------------------------------------------------------------ #

    def _find_optimal_trigger_timing(self, estimated_phases: np.ndarray, reference_time: float, target_phase:float,) -> Optional[Dict[str, float]]:
        """"
        Find optimal trigger timing based on estimated phases.
        
        Args:
            estimated_phases: Array of estimated phase values
            reference_time: Current timestamp
            target_phase
            
        Returns:
            Dictionary with trigger timing or None if no suitable timing found
        """

        self.last_phase_error = None

        num_samples           = estimated_phases.shape[0]
        future_phases         = estimated_phases[num_samples // 2:]
        future_phases         = future_phases[:self.max_future_samples]

        phase_diffs           = np.angle(np.exp(1j * (future_phases - target_phase)))
        best_idx              = np.argmin(np.abs(phase_diffs))
        phase_error           = np.abs(phase_diffs[best_idx])
        self.last_phase_error = phase_error

        if phase_error > self.phase_tolerance:
            print("Phase error is too large, skipping trigger")
            return None

        time_offset = (best_idx * self.downsample_ratio) / self.sampling_frequency
        if time_offset < MINIMUM_TRIGGER_DELAY_SECONDS:
            print("Trigger delay is too short, skipping trigger")
            return None

        print(f'Trigger scheduled {time_offset:.3f}s from now', flush=True)
        return {'trigger_offset': time_offset}

    # ------------------------------------------------------------------ #
    #  MEP & reward processing                                           #
    # ------------------------------------------------------------------ #

    def process_pulse(
            self, reference_time: float, reference_index: int, time_offsets: np.ndarray,
            eeg_buffer: np.ndarray, emg_buffer: np.ndarray, is_coil_at_target: bool, stage_name: str, trial_in_stage: int) -> dict[str, Any] | None:
        """
        For each MEP buffer:
          1. Append to MEP history and compute reward.
          2. Fill in the most recent unfilled log entry.
          3. If in RL block and a pending experience exists, build the
             next-state observation sequence and push to replay, then
             run one gradient step and decay epsilon.
        """
        isi = np.random.uniform(2, 3)
        self.earliest_next_pulse_time = reference_time + isi

        start_time = time.time()

        print("Processing pulse...")
        #       - Check how MEP is calculated
        #mep = np.max(np.abs(emg_buffer[:, 0]))
        #mep = np.ptp(emg_buffer[:, 0])
        #mep = np.max(emg_buffer[:, 0]) - np.min(emg_buffer[:, 0])
        sample_mask = (time_offsets >= 0.025) & (time_offsets <= 0.05)
        emg_window = emg_buffer[sample_mask, :]
        mep = np.ptp(emg_window[:, 0])
        print(f"EMG buffer shape: {emg_buffer.shape}, time_offsets: min={time_offsets[0]:.4f} max={time_offsets[-1]:.4f}")  

        self._last_mep = mep
        self._mep_history.append(mep)

        reward            = self._compute_reward(mep)
        self._last_reward = reward

        # Fill in the most recent log entry that is missing a MEP
        for entry in reversed(self._log):
            if entry["mep"] is None:
                entry["mep"] = mep
                entry["reward"] = reward
                entry["isi"] = isi
                entry["earliest_next_pulse_time"] = self.earliest_next_pulse_time
                break

        # RL training step (only in the RL block, and only if a pending experience exists)
        if (
            self._blocks[self._block_idx]["mode"] == "rl"
            and self._pending_obs    is not None
            and self._pending_action is not None
            and self._pending_obs_seq is not None
        ):
            self._action_counts[self._pending_action] += 1   # ← move here
            self._action_rewards[self._pending_action] += reward  # ← move here
            # Build next_obs and next_obs_seq
            next_obs = np.array([
                ACTION_TO_PHASE.get(self._pending_action, 0.0),
                mep,
            ])
            next_window = deque(
                list(self._pending_obs_seq), maxlen=SEQUENCE_LENGTH
            )
            next_window.append(next_obs)
            next_obs_seq = np.array(list(next_window))

            # This pulse completes trial `trial_in_stage` (0-based) of the RL stage.
            done = (trial_in_stage + 1 >= self._blocks[self._block_idx]["n_trials"])

            loss = self._agent.store_and_train(
                obs_seq = self._pending_obs_seq,
                action = self._pending_action,
                reward = reward,
                next_obs_seq = next_obs_seq,
                done = done,
            )
            print(f"Loss: {loss}")

            self._agent.decay_epsilon()
            self._check_episode_boundary()

            if loss is not None:
                print(
                    f"[RL] loss={loss:.4f} | MEP={mep:.1f}µV | "
                    f"reward={reward:.2f} | ε={self._agent.epsilon:.3f}"
                )

        # Clear pending regardless of block (avoids stale data)
        self._pending_obs     = None
        self._pending_action  = None
        self._pending_obs_seq = None
        
        print(f"Pulse processed in {time.time() - start_time:.3f} seconds")


    def _compute_reward(self, current_mep: float) -> float:
        """
        Mirrors MATLAB POPSTAR reward function:

            weights  = linspace(0.5, 1.5, N)   [N = number of MEPs so far]
            weights /= sum(weights)             [normalise to sum=1]
            avg_MEP  = dot(weights, history)    [recency-weighted average]
            reward   = current_MEP − avg_MEP × 1.2

        more weight to recent trials
        positive reward if the current MEP exceeds 120 % of recent weighted average.
        """
        n = len(self._mep_history)
        if n > 1:
            weights  = np.linspace(0.5, 1.5, n)
            weights /= weights.sum()
            avg_mep  = float(np.dot(weights, np.array(self._mep_history)))
        else:
            # First trial: no history yet, baseline equals the current MEP
            # → reward = current_mep − current_mep × 1.2 = −0.2 × current_mep
            avg_mep = current_mep

        reward = current_mep - avg_mep * 1.2
        return float(np.clip(reward, -REWARD_CAP, REWARD_CAP))

    # ------------------------------------------------------------------ #
    #  Internal: block & episode management                               #
    # ------------------------------------------------------------------ #

    def _trials_before_current_block(self) -> int:
        """Number of trials in the blocks preceding the current one."""
        return sum(block["n_trials"] for block in self._blocks[:self._block_idx])

    def _start_block_for_stage(self, stage_name: str) -> bool:
        """
        Bind block state to a protocol stage reported by the coordinator.

        The first stage seen starts block 0; every later stage advances one block.
        Returns False if the protocol has more stages than the decider has blocks.

        Assumes stage names in the protocol are unique, which they are in popstar.yaml
        (phase_1, phase_2, phase_3).
        """
        previous_stage_name = self._current_stage_name
        self._current_stage_name = stage_name

        if previous_stage_name is None:
            print(
                f"\n[Decider] ===== Starting block: {self._blocks[0]['name']} "
                f"(stage '{stage_name}') =====\n"
            )
            return True

        return self._advance_block(stage_name)

    def _advance_block(self, stage_name: str) -> bool:
        """Move to the block for the newly started stage. Returns False if out of blocks."""
        self._save_all()

        # Freeze the best RL phase before leaving the RL block.
        if self._blocks[self._block_idx]["mode"] == "rl":
            avg_rewards = np.divide(
                self._action_rewards[1:],
                self._action_counts[1:],
                out=np.zeros(N_ACTIONS),
                where=self._action_counts[1:] > 0
            )
            best_action = int(np.argmax(avg_rewards)) + 1
            self._best_rl_phase = ACTION_TO_PHASE[best_action]
            print(f"[Decider] Best RL phase: action={best_action}, phase={np.degrees(self._best_rl_phase):.1f}°")

        self._block_idx += 1

        if self._block_idx >= len(self._blocks):
            print(
                f"[Decider] Stage '{stage_name}' started but all blocks are done — "
                f"the protocol has more stages than the decider has blocks."
            )
            return False

        self._mep_history      = []
        self._pending_obs      = None
        self._pending_action   = None
        self._pending_obs_seq  = None

        if self._blocks[self._block_idx]["mode"] == "rl":
            self._agent.reset_episode()

        print(
            f"\n[Decider] ===== Starting block: {self._blocks[self._block_idx]['name']} "
            f"(stage '{stage_name}') =====\n"
        )
        return True

    def _check_episode_boundary(self) -> None:
        """
        Save progress every 40 pulses and reset its LSTM 
        """
        self._episode_step  += 1
        self._episode_total += self._last_reward

        if self._episode_step >= self._steps_per_episode:
            self._episode_num += 1
            self._episode_rewards.append(self._episode_total)
            print(
                f"[RL] Episode {self._episode_num} complete | "
                f"total_reward={self._episode_total:.2f} | "
                f"ε={self._agent.epsilon:.3f}"
            )
            self._episode_step  = 0
            self._episode_total = 0.0
            self._agent.reset_episode()

            agent_path = os.path.join(SAVE_PATH, f"{self.subject_id}_dqn_agent.pt")
            self._agent.save(agent_path)

