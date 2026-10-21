"""
POPSTAR RL training decider
===========================
Phastimate (real-time EEG phase estimation) + a DQN+LSTM agent that learns which of 8
discrete EEG phase bins maximises MEP amplitude. This decider ONLY does the RL block.

At the end of every episode, at the final trial, and on exit it writes
    data/<subject>_best_rl_phase.json
which the post-test deciders (04b, 04c) load. Run this one first.

References:
  Zrenner et al. (2020) NeuroImage 214, 116761  — Phastimate algorithm
  POPSTAR 2025 MATLAB codebase — RL environment & reward design
"""
import json
import os
import sys
import random
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import hilbert

from popstar_common import (
    ACTION_TO_PHASE, N_ACTIONS, SAVE_PATH, PhastimateDecider, best_phase_path,
)

# --- Session structure ---
# Must match the number of trials in the RL stage of the protocol yaml.
N_RL_TRIALS = 800

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
class Decider(PhastimateDecider):
    """
    RL-guided TMS phase targeting. The agent picks one of 8 phases each trial and learns
    from the MEP feedback. Single block; the protocol should have one stage of N_RL_TRIALS.
    """

    BLOCK_LABEL = "rl_training"
    BLOCK_NAME = "RL training"
    BLOCK_INDEX = 1

    def __init__(self, subject_id: int, num_eeg_channels: int, num_emg_channels,
                 sampling_frequency: float, runtime_params):
        super().__init__(subject_id, num_eeg_channels, num_emg_channels, sampling_frequency, runtime_params)
        self.n_trials = N_RL_TRIALS

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("GPU is available, using GPU" if self.device.type == "cuda" else "GPU is not available, using CPU")

        # Per-action statistics used to pick the best phase (index 1-8)
        self._action_counts = np.zeros(N_ACTIONS + 1)
        self._action_rewards = np.zeros(N_ACTIONS + 1)   # cumulative reward per action
        self._action_meps = np.zeros(N_ACTIONS + 1)      # cumulative MEP per action

        # RL components
        self._agent = DQNAgent(self.device)
        self._mep_history: List[float] = []

        # Pending experience: set when a trigger is scheduled, consumed when the MEP arrives
        self._pending_obs:     Optional[np.ndarray] = None
        self._pending_action:  Optional[int]        = None
        self._pending_obs_seq: Optional[np.ndarray] = None

        self._last_reward = 0.0

        # RL episode tracking
        self._steps_per_episode = 40
        self._episode_step = 0
        self._episode_num = 0
        self._episode_rewards: List[float] = []
        self._episode_total = 0.0

        # Resume from a previous run for this subject, if any
        agent_path = os.path.join(SAVE_PATH, f"{self.subject_id}_dqn_agent.pt")
        if os.path.isfile(agent_path):
            self._agent.load(agent_path)
        self._load_action_stats()

        print(f"[Decider] Initialised | device={self.device} | RL trials={N_RL_TRIALS} | Subject ID={self.subject_id}")

    # ------------------------------------------------------------------ #
    #  Phase selection                                                    #
    # ------------------------------------------------------------------ #
    def _select_target_phase(self, estimated_phases: np.ndarray, trial_in_stage: int) -> Optional[float]:
        """
        Agent selects one of 8 discrete phases based on the current EEG phase and the most
        recent MEP amplitude. The (obs, action, obs_window_snapshot) triple is stored as
        pending so the reward can be assigned after the MEP arrives.
        """
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

    def _trial_info(self, trial_in_stage: int) -> Dict[str, Any]:
        return {"mode": "rl", "action": self._pending_action}

    # ------------------------------------------------------------------ #
    #  MEP & reward processing                                            #
    # ------------------------------------------------------------------ #
    def _on_mep(self, mep: float, trial_in_stage: int) -> Dict[str, Any]:
        """
        1. Append MEP to history and compute reward.
        2. If a pending experience exists: update per-action stats, push to replay, run one
           gradient step, decay epsilon, check the episode boundary.
        """
        self._mep_history.append(mep)
        reward = self._compute_reward(mep)
        self._last_reward = reward

        if (
            self._pending_obs     is not None
            and self._pending_action is not None
            and self._pending_obs_seq is not None
        ):
            self._action_counts[self._pending_action] += 1
            self._action_rewards[self._pending_action] += reward
            self._action_meps[self._pending_action] += mep

            next_obs = np.array([ACTION_TO_PHASE.get(self._pending_action, 0.0), mep])
            next_window = deque(list(self._pending_obs_seq), maxlen=SEQUENCE_LENGTH)
            next_window.append(next_obs)
            next_obs_seq = np.array(list(next_window))

            # This pulse completes trial `trial_in_stage` (0-based) of the RL stage.
            done = (trial_in_stage + 1 >= N_RL_TRIALS)

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

            if done:
                print("[Decider] Final RL trial complete — saving agent and best phase.")
                self._save_all()

        # Clear pending (avoids stale data)
        self._pending_obs     = None
        self._pending_action  = None
        self._pending_obs_seq = None

        return {"reward": reward}

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
    #  Best phase                                                         #
    # ------------------------------------------------------------------ #
    def _best_action(self) -> Optional[int]:
        """Action (1-8) with the highest mean reward. Untried actions are excluded
        (rewards are mostly negative by construction, so an untried action's 0 would win)."""
        counts = self._action_counts[1:]
        if not np.any(counts > 0):
            return None
        avg = np.full(N_ACTIONS, -np.inf)
        np.divide(self._action_rewards[1:], counts, out=avg, where=counts > 0)
        return int(np.argmax(avg)) + 1

    def _save_best_phase(self) -> None:
        action = self._best_action()
        if action is None:
            return
        counts = self._action_counts[1:]
        rewards = self._action_rewards[1:]
        meps = self._action_meps[1:]
        mean_of = lambda s: [float(s[i] / counts[i]) if counts[i] > 0 else None for i in range(N_ACTIONS)]
        payload = {
            "subject_id": self.subject_id,
            "best_action": action,
            "best_phase_rad": float(ACTION_TO_PHASE[action]),
            "best_phase_deg": float(np.degrees(ACTION_TO_PHASE[action])),
            "n_rl_trials_completed": int(counts.sum()),
            "action_phases_rad": [float(ACTION_TO_PHASE[k]) for k in range(1, N_ACTIONS + 1)],
            "action_counts": counts.tolist(),
            "action_reward_sums": rewards.tolist(),
            "action_mep_sums": meps.tolist(),
            "mean_reward_per_action": mean_of(rewards),
            "mean_mep_per_action": mean_of(meps),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(SAVE_PATH, exist_ok=True)
        path = best_phase_path(self.subject_id)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Decider] Best RL phase: action={action}, phase={payload['best_phase_deg']:.1f}° → {path}")

    def _load_action_stats(self) -> None:
        """Restore per-action stats so a restarted session keeps its history."""
        path = best_phase_path(self.subject_id)
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                d = json.load(f)
            self._action_counts[1:]  = d["action_counts"]
            self._action_rewards[1:] = d["action_reward_sums"]
            self._action_meps[1:]    = d["action_mep_sums"]
            print(f"[Decider] Restored action stats ({int(self._action_counts.sum())} earlier RL trials) from {path}")
        except Exception as exc:
            print(f"[Decider] Could not restore action stats from {path}: {exc}")

    # ------------------------------------------------------------------ #
    #  Episodes & saving                                                  #
    # ------------------------------------------------------------------ #
    def _check_episode_boundary(self) -> None:
        """Every `_steps_per_episode` pulses: save progress and reset the LSTM."""
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
            self._save_all()

    def _save_all(self) -> None:
        super()._save_all()   # trial log
        if not hasattr(self, "_agent"):
            return
        os.makedirs(SAVE_PATH, exist_ok=True)
        self._agent.save(os.path.join(SAVE_PATH, f"{self.subject_id}_dqn_agent.pt"))
        self._save_best_phase()

        if self._episode_rewards:
            import csv
            ep_path = os.path.join(SAVE_PATH, f"{self.subject_id}_episode_rewards.csv")
            with open(ep_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "total_reward"])
                for i, r in enumerate(self._episode_rewards, 1):
                    writer.writerow([i, r])
            print(f"[Decider] Episode rewards saved → {ep_path}")
