#!/usr/bin/env python3
"""
human_balance_evaluation_fast.py

Fast evaluation script for pre-trained human balance GAIL models.
Focuses on essential metrics and skips time-consuming spectrum/PCA/t-SNE analysis.
"""

import os
import random
import math
from copy import deepcopy
from pathlib import Path
import json
import time
import csv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
from fastdtw import fastdtw
import gymnasium as gym
from gymnasium import spaces

# ====================== CONFIGURATION ======================
CSV_PATH = "all_excel_measurements.csv"
GROUP_COL = "fn_index"
TIME_COL = "n"
X_COL = "X"
Y_COL = "Y"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Environment parameters
DT = 0.05
MAX_THETA = np.radians(12)
MAX_PHI = np.radians(12)
MAX_ANG_VEL = 15.0
GRAVITY = 9.81
MASS = 71.83
LENGTH = 0.85
MAX_TORQUE = 75.0
INERTIA = MASS * LENGTH**2 / 3.0
DAMPING = 4.0

# Evaluation parameters
MAX_STEPS_EVALUATION = 15000

# ====================== UTILITIES ======================

def to_torch(x, device=DEVICE, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)

def _json_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return str(obj)

# ====================== ENVIRONMENT ======================

class HumanBalanceEnv(gym.Env):
    def __init__(self, dt=DT, max_steps=600, trajectories=None, trajectory_prob=0.3,
                 enable_noise=True, enable_delay=True, action_threshold=0.01,
                 discrete_actions=True, survival_bonus=1.0,
                 angle_reward_scale=3.0, angle_reward_sigma=np.radians(1.0),
                 vel_penalty_weight=0, torque_penalty=1e-2,
                 action_change_penalty=0, max_noise=1.5, max_delay=0.4):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.dt = dt
        self.max_steps = max_steps
        self.state = np.zeros(4, dtype=np.float32)
        self.step_count = 0
        self.trajectories = trajectories if trajectories is not None else []
        self.trajectory_prob = trajectory_prob
        self.enable_noise = enable_noise
        self.enable_delay = enable_delay
        self.max_noise = max_noise
        self.max_delay = max_delay
        self.current_noise = 0.0
        self.current_delay = 0.0
        self.last_torque = np.zeros(2, dtype=np.float32)
        self.termination_reason = None

        self.action_threshold = action_threshold
        self.discrete_actions = discrete_actions
        self.survival_bonus = survival_bonus
        self.angle_reward_scale = angle_reward_scale
        self.angle_reward_sigma = angle_reward_sigma
        self.vel_penalty_weight = vel_penalty_weight
        self.torque_penalty = torque_penalty
        self.action_change_penalty = action_change_penalty

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        progress = np.random.rand()
        self.current_noise = progress * self.max_noise if self.enable_noise else 0.0
        self.current_delay = progress * self.max_delay if self.enable_delay else 0.0
        if self.trajectories and np.random.rand() < self.trajectory_prob:
            self._reset_from_trajectory()
        else:
            self._random_reset()
        self.step_count = 0
        self.last_torque = np.zeros(2, dtype=np.float32)
        self.termination_reason = None
        return self._get_observation(), {}

    def _discretize_action(self, action):
        if not self.discrete_actions:
            return action
        discrete_action = np.zeros_like(action)
        for i in range(len(action)):
            if action[i] > self.action_threshold:
                discrete_action[i] = 1.0
            elif action[i] < -self.action_threshold:
                discrete_action[i] = -1.0
            else:
                discrete_action[i] = 0.0
        return discrete_action

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        discrete_action = self._discretize_action(a)
        torque_theta = discrete_action[0] * MAX_TORQUE if self.discrete_actions else np.clip(a[0], -1, 1) * MAX_TORQUE
        torque_phi = discrete_action[1] * MAX_TORQUE if self.discrete_actions else np.clip(a[1], -1, 1) * MAX_TORQUE
        torques = np.array([torque_theta, torque_phi], dtype=np.float32)

        θ, φ, dθ, dφ = self.state
        θ_acc = -(GRAVITY/LENGTH)*math.sin(θ) + torque_theta/INERTIA - DAMPING * dθ
        φ_acc = -(GRAVITY/LENGTH)*math.sin(φ) + torque_phi/INERTIA - DAMPING * dφ
        new_dθ = float(np.clip(dθ + θ_acc*self.dt, -MAX_ANG_VEL, MAX_ANG_VEL))
        new_dφ = float(np.clip(dφ + φ_acc*self.dt, -MAX_ANG_VEL, MAX_ANG_VEL))
        new_θ = float(θ + new_dθ*self.dt)
        new_φ = float(φ + new_dφ*self.dt)

        terminated_theta = abs(new_θ) > MAX_THETA
        terminated_phi = abs(new_φ) > MAX_PHI
        terminated = terminated_theta or terminated_phi

        if terminated:
            if terminated_theta and terminated_phi:
                self.termination_reason = 'both_angles'
            elif terminated_theta:
                self.termination_reason = 'theta'
            else:
                self.termination_reason = 'phi'

        env_reward = self.compute_reward(new_θ, new_φ, new_dθ, new_dφ, torques, terminated)

        self.state = np.array([new_θ, new_φ, new_dθ, new_dφ], dtype=np.float32)
        self.last_torque = torques.copy()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        if truncated:
            self.termination_reason = 'time_limit'
        return self._get_observation(), float(env_reward), bool(terminated), bool(truncated), {}

    def compute_reward(self, θ, φ, dθ, dφ, torques, terminated):
        ang2 = θ*θ + φ*φ
        angle_reward = self.angle_reward_scale * math.exp(-ang2 / (2 * (self.angle_reward_sigma**2 + 1e-12)))
        vel_penalty = self.vel_penalty_weight * (dθ**2 + dφ**2)
        torque_pen = self.torque_penalty * (((torques[0] / MAX_TORQUE)**2) + ((torques[1] / MAX_TORQUE)**2))
        action_change = self.action_change_penalty * float(np.linalg.norm(torques - self.last_torque))
        survival = float(self.survival_bonus) if not terminated else -5.0
        r = survival + angle_reward - vel_penalty - torque_pen - action_change
        return r

    def _reset_from_trajectory(self):
        traj = random.choice(self.trajectories)
        if len(traj) < 2:
            self._random_reset()
            return
        idx = np.random.randint(0, len(traj)-1)
        x, y = traj[idx]; nx, ny = traj[idx+1]
        θ = float(math.asin(np.clip(x/LENGTH, -1.0, 1.0)))
        φ = float(math.asin(np.clip(y/LENGTH, -1.0, 1.0)))
        next_θ = float(math.asin(np.clip(nx/LENGTH, -1.0, 1.0)))
        next_φ = float(math.asin(np.clip(ny/LENGTH, -1.0, 1.0)))
        dθ = float((next_θ - θ)/self.dt)
        dφ = float((next_φ - φ)/self.dt)
        self.state = np.array([θ, φ, np.clip(dθ, -MAX_ANG_VEL, MAX_ANG_VEL), np.clip(dφ, -MAX_ANG_VEL, MAX_ANG_VEL)], dtype=np.float32)

    def _random_reset(self):
        self.state = np.array([
            np.random.uniform(-np.radians(0.5), np.radians(0.5)),
            np.random.uniform(-np.radians(0.5), np.radians(0.5)),
            0.0, 0.0
        ], dtype=np.float32)

    def _get_observation(self):
        θ, φ, dθ, dφ = self.state
        noisy_θ = θ + math.radians(np.random.normal(0, self.current_noise)) if self.enable_noise else θ
        noisy_φ = φ + math.radians(np.random.normal(0, self.current_noise)) if self.enable_noise else φ
        θ_scaled = np.clip(noisy_θ / MAX_THETA, -1.0, 1.0)
        φ_scaled = np.clip(noisy_φ / MAX_PHI, -1.0, 1.0)
        dθ_scaled = np.clip(dθ / MAX_ANG_VEL, -1.0, 1.0)
        dφ_scaled = np.clip(dφ / MAX_ANG_VEL, -1.0, 1.0)
        return np.array([
            θ_scaled, φ_scaled,
            math.cos(noisy_θ), math.sin(noisy_θ),
            math.cos(noisy_φ), math.sin(noisy_φ),
            dθ_scaled, dφ_scaled,
            self.current_delay / self.max_delay,
            self.current_noise / self.max_noise
        ], dtype=np.float32)

# ====================== NETWORKS ======================

def mlp(in_dim, out_dim, hidden=(256,256), activation=nn.ReLU, dropout=0.0):
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        layers.append(nn.LayerNorm(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

class PolicyValue(nn.Module):
    def __init__(self, obs_dim=10, act_dim=2, hidden=(256,256)):
        super().__init__()
        self.actor = nn.Sequential(mlp(obs_dim, act_dim, hidden), nn.Tanh())
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))
        self.critic = mlp(obs_dim, 1, hidden)

    def forward(self, obs):
        mu = self.actor(obs)
        std = torch.exp(self.log_std)
        return mu, std

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

# ====================== DATA LOADING ======================

def load_trajectories(csv_path, group_col='path', time_col='n', x_col='X', y_col='y'):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path, encoding='utf-8')
    df = df[df['tp'] == "ROMBERG"].copy()

    for c in [x_col, y_col, time_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=[group_col, time_col, x_col, y_col], inplace=True)

    groups = []
    labels = []

    print(f"Grouping by '{group_col}' to create trajectories...")

    for name, g in df.groupby(group_col):
        arr = g.sort_values(time_col)[[x_col, y_col]].values / 1000.0

        if len(arr) >= 100:
            groups.append(arr)
            participant = g['name'].iloc[0] if 'name' in g.columns else 'Unknown'
            test_type = g['tp'].iloc[0] if 'tp' in g.columns else 'Unknown'
            sensitivity = g['fn_sens'].iloc[0] if 'fn_sens' in g.columns else 'Unknown'
            labels.append(f"{participant}_{test_type}_sens{sensitivity}")

    print(f"Loaded {len(groups)} trajectories from {len(df[group_col].unique())} unique test sessions")
    return groups, labels

def positions_to_states(traj_xy):
    xs, ys = traj_xy[:, 0], traj_xy[:, 1]
    thetas = np.arcsin(np.clip(xs / LENGTH, -1.0, 1.0))
    phis = np.arcsin(np.clip(ys / LENGTH, -1.0, 1.0))
    dtheta = np.zeros_like(thetas)
    dphi = np.zeros_like(phis)
    for i in range(1, len(thetas)-1):
        dtheta[i] = (thetas[i+1] - thetas[i-1])/(2*DT)
        dphi[i] = (phis[i+1] - phis[i-1])/(2*DT)
    if len(thetas) > 1:
        dtheta[0] = (thetas[1] - thetas[0]) / DT
        dphi[0] = (phis[1] - phis[0]) / DT
        dtheta[-1] = (thetas[-1] - thetas[-2]) / DT
        dphi[-1] = (phis[-1] - phis[-2]) / DT
    dtheta = np.clip(dtheta, -MAX_ANG_VEL, MAX_ANG_VEL)
    dphi = np.clip(dphi, -MAX_ANG_VEL, MAX_ANG_VEL)
    return np.stack([thetas, phis, dtheta, dphi], axis=1)

# ====================== TRAJECTORY SAVING ======================

def save_expert_trajectories_csv(trajectories, labels, filename="expert_trajectories.csv"):
    print(f"Saving expert trajectories to {filename}...")
    
    all_data = []
    for traj_idx, (traj, label) in enumerate(zip(trajectories, labels)):
        states = positions_to_states(traj)
        
        for point_idx, (point, state) in enumerate(zip(traj, states)):
            x, y = point
            theta, phi, dtheta, dphi = state
            
            all_data.append({
                'trajectory_id': traj_idx,
                'trajectory_label': label,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(trajectories)} expert trajectories with {len(df)} total points to {filename}")
    return df

def save_agent_trajectories_csv(agent_trajectories, filename="agent_trajectories.csv"):
    print(f"Saving agent trajectories to {filename}...")
    
    all_data = []
    for traj_idx, states in enumerate(agent_trajectories):
        for point_idx, state in enumerate(states):
            theta, phi, dtheta, dphi = state
            x = LENGTH * math.sin(theta)
            y = LENGTH * math.sin(phi)
            
            all_data.append({
                'trajectory_id': traj_idx,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(agent_trajectories)} agent trajectories with {len(df)} total points to {filename}")
    return df

def save_comparison_trajectories_csv(expert_trajectories, expert_labels, agent_trajectories, 
                                   filename="comparison_trajectories.csv"):
    print(f"Saving comparison trajectories to {filename}...")
    
    all_data = []
    
    # Save expert trajectories
    for traj_idx, (traj, label) in enumerate(zip(expert_trajectories, expert_labels)):
        states = positions_to_states(traj)
        
        for point_idx, (point, state) in enumerate(zip(traj, states)):
            x, y = point
            theta, phi, dtheta, dphi = state
            
            all_data.append({
                'source': 'expert',
                'trajectory_id': traj_idx,
                'trajectory_label': label,
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })
    
    # Save agent trajectories  
    for traj_idx, states in enumerate(agent_trajectories):
        for point_idx, state in enumerate(states):
            theta, phi, dtheta, dphi = state
            x = LENGTH * math.sin(theta)
            y = LENGTH * math.sin(phi)
            
            all_data.append({
                'source': 'agent',
                'trajectory_id': traj_idx,
                'trajectory_label': f'agent_{traj_idx}',
                'point_index': point_idx,
                'time_seconds': point_idx * DT,
                'x_coordinate': x,
                'y_coordinate': y,
                'theta_radians': theta,
                'phi_radians': phi,
                'dtheta_radians_per_sec': dtheta,
                'dphi_radians_per_sec': dphi
            })
    
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(expert_trajectories)} expert + {len(agent_trajectories)} agent trajectories to {filename}")
    return df

# ====================== AGENT TRAJECTORY COLLECTION ======================

def collect_agent_trajectories(policy, env, n_episodes=10, max_steps=15000):
    trajs = []
    survival_steps = []
    termination_reasons = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        states = []
        done = False
        steps = 0
        while not done and steps < max_steps:
            obs_t = to_torch(obs[None,:])
            with torch.no_grad():
                mu, _ = policy(obs_t)
                act = mu.cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(act)
            states.append(env.state.copy())
            done = terminated or truncated
            steps += 1
        
        if len(states) > 0:
            trajs.append(np.array(states, dtype=np.float32))
            survival_steps.append(len(states))
            termination_reasons.append(env.termination_reason)

    if trajs:
        traj_lengths = [len(traj) for traj in trajs]
        print(f"Collected {len(trajs)} agent trajectories:")
        print(f"  Lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
        
        # Analyze termination reasons
        reason_counts = {}
        for reason in termination_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        print("  Termination reasons:")
        for reason, count in reason_counts.items():
            print(f"    {reason}: {count} ({count/len(termination_reasons)*100:.1f}%)")
    else:
        print("No agent trajectories collected.")

    return trajs

# ====================== MODEL LOADING ======================

def load_best_model(checkpoint_dir="checkpoints_stable", obs_dim=10, act_dim=2):
    policy_path = Path(checkpoint_dir) / "policy_best.pt"

    if not policy_path.exists():
        raise FileNotFoundError(f"Best policy model not found at {policy_path}")

    policy = PolicyValue(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    policy.eval()

    print(f"Loaded best model from {policy_path}")
    
    # Try to load training metrics if available
    metrics_path = Path(checkpoint_dir) / "metrics_best.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if "best_combined_reward" in metrics and len(metrics["best_combined_reward"]) > 0:
                best_reward = metrics["best_combined_reward"][-1]
                best_iter = metrics["iter"][-1] if "iter" in metrics else "unknown"
                print(f"📊 Best model performance: reward {best_reward:.4f} at iteration {best_iter}")
        except Exception as e:
            print(f"Note: Could not load best metrics: {e}")

    return policy

# ====================== FAST METRICS ======================

def path_length(theta_phi):
    """Calculate total path length in angle space"""
    if theta_phi.shape[0] < 2:
        return 0.0
    diffs = np.diff(theta_phi, axis=0)
    steps = np.sqrt((diffs**2).sum(axis=1))
    return float(steps.sum())

def sway_area_convex_hull(theta_phi):
    """Calculate sway area using convex hull method"""
    if theta_phi.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(theta_phi)
        return float(hull.volume)
    except Exception:
        return 0.0

def compute_basic_stats(states):
    """Compute basic statistics for trajectory states"""
    stats = {}
    for i, name in enumerate(['theta', 'phi', 'dtheta', 'dphi']):
        sig = states[:, i]
        stats[name] = {
            'mean': float(np.mean(sig)),
            'std': float(np.std(sig)),
            'max': float(np.max(sig)),
            'min': float(np.min(sig)),
            'range': float(np.ptp(sig))
        }
    return stats

def compute_reproducibility_metrics(list_of_states, verbose=True):
    """Fast reproducibility metrics without spectral analysis"""
    valid_trajectories = [traj for traj in list_of_states if len(traj) >= 2]

    if len(valid_trajectories) < 2:
        return {
            'n_traj': len(valid_trajectories),
            'combined_score': 0.0
        }

    try:
        path_lengths = [path_length(traj[:, :2]) for traj in valid_trajectories]
        sway_areas = [sway_area_convex_hull(traj[:, :2]) for traj in valid_trajectories]

        path_length_cv = np.std(path_lengths) / (np.mean(path_lengths) + 1e-12)
        sway_area_cv = np.std(sway_areas) / (np.mean(sway_areas) + 1e-12)

        combined_score = 1.0 / (1.0 + path_length_cv + sway_area_cv)

        out = {
            'n_traj': len(valid_trajectories),
            'combined_score': float(combined_score),
            'path_length_mean': float(np.mean(path_lengths)),
            'path_length_std': float(np.std(path_lengths)),
            'sway_area_mean': float(np.mean(sway_areas)),
            'sway_area_std': float(np.std(sway_areas))
        }

        if verbose:
            print(f"Reproducibility for {len(valid_trajectories)} trajectories:")
            print(f"  Combined score: {combined_score:.4f}")
            print(f"  Path length: {np.mean(path_lengths):.3f} ± {np.std(path_lengths):.3f}")
            print(f"  Sway area: {np.mean(sway_areas):.6f} ± {np.std(sway_areas):.6f}")
        return out

    except Exception as e:
        print(f"Error computing reproducibility metrics: {e}")
        return {
            'n_traj': len(valid_trajectories),
            'combined_score': 0.0
        }

def compute_fast_comparison_metrics(expert_trajs, agent_trajs):
    """Fast comparison metrics without PCA/t-SNE"""
    metrics = {}
    
    # Basic trajectory statistics
    if expert_trajs and agent_trajs:
        expert_lengths = [len(traj) for traj in expert_trajs]
        agent_lengths = [len(traj) for traj in agent_trajs]
        
        metrics['expert_mean_length'] = float(np.mean(expert_lengths))
        metrics['agent_mean_length'] = float(np.mean(agent_lengths))
        metrics['length_ratio'] = float(metrics['agent_mean_length'] / metrics['expert_mean_length'])
        
        # Angle range comparison
        expert_theta_ranges = [np.ptp(traj[:, 0]) for traj in expert_trajs]
        agent_theta_ranges = [np.ptp(traj[:, 0]) for traj in agent_trajs]
        expert_phi_ranges = [np.ptp(traj[:, 1]) for traj in expert_trajs]
        agent_phi_ranges = [np.ptp(traj[:, 1]) for traj in agent_trajs]
        
        metrics['expert_theta_range'] = float(np.mean(expert_theta_ranges))
        metrics['agent_theta_range'] = float(np.mean(agent_theta_ranges))
        metrics['expert_phi_range'] = float(np.mean(expert_phi_ranges))
        metrics['agent_phi_range'] = float(np.mean(agent_phi_ranges))
        
        # Velocity comparison
        expert_vel = [np.mean(np.sqrt(traj[:, 2]**2 + traj[:, 3]**2)) for traj in expert_trajs]
        agent_vel = [np.mean(np.sqrt(traj[:, 2]**2 + traj[:, 3]**2)) for traj in agent_trajs]
        metrics['expert_mean_velocity'] = float(np.mean(expert_vel))
        metrics['agent_mean_velocity'] = float(np.mean(agent_vel))
    
    return metrics

# ====================== VISUALIZATION ======================

def plot_trajectory_comparison(expert_trajs, agent_trajs, n_examples=3):
    """Plot side-by-side comparison of expert vs agent trajectories"""
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 4*n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        if i < len(expert_trajs):
            # Expert trajectory
            expert_states = positions_to_states(expert_trajs[i])
            expert_theta = expert_states[:, 0]
            expert_phi = expert_states[:, 1]
            time_expert = np.arange(len(expert_theta)) * DT
            
            axes[i, 0].plot(time_expert, expert_theta, 'b-', label='Theta', alpha=0.7)
            axes[i, 0].plot(time_expert, expert_phi, 'r-', label='Phi', alpha=0.7)
            axes[i, 0].set_title(f'Expert Trajectory {i+1}')
            axes[i, 0].set_xlabel('Time (s)')
            axes[i, 0].set_ylabel('Angle (rad)')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
        
        if i < len(agent_trajs):
            # Agent trajectory
            agent_theta = agent_trajs[i][:, 0]
            agent_phi = agent_trajs[i][:, 1]
            time_agent = np.arange(len(agent_theta)) * DT
            
            axes[i, 1].plot(time_agent, agent_theta, 'b-', label='Theta', alpha=0.7)
            axes[i, 1].plot(time_agent, agent_phi, 'r-', label='Phi', alpha=0.7)
            axes[i, 1].set_title(f'Agent Trajectory {i+1}')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Angle (rad)')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_survival_analysis(agent_trajs):
    """Plot survival step distribution"""
    if not agent_trajs:
        return
        
    survival_steps = [len(traj) for traj in agent_trajs]
    plt.figure(figsize=(10, 6))
    plt.hist(survival_steps, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(survival_steps), color='red', linestyle='--', 
                label=f'Mean: {np.mean(survival_steps):.1f} steps')
    plt.xlabel('Survival Steps')
    plt.ylabel('Frequency')
    plt.title('Agent Survival Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('survival_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

# ====================== MAIN EVALUATION FUNCTION ======================

def main_evaluation():
    """Fast evaluation function using pre-trained models"""
    # Set seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        print("=== Human Balance GAIL Evaluation (Fast) ===")
        print(f"Device: {DEVICE}")
        print(f"Seed: {SEED}")

        # Load and prepare data
        print("\n1. Loading trajectories...")
        all_trajs, labels = load_trajectories(CSV_PATH, GROUP_COL, TIME_COL, X_COL, Y_COL)

        # Use all trajectories for evaluation
        print(f"Using {len(all_trajs)} trajectories for evaluation")

        # Load pre-trained model
        print("\n2. Loading pre-trained model...")
        obs_dim = 10  # Default observation dimension
        act_dim = 2   # Default action dimension
        
        policy = load_best_model("checkpoints_stable", obs_dim, act_dim)

        # Create evaluation environment
        print("\n3. Creating evaluation environment...")
        eval_env = HumanBalanceEnv(
            trajectories=all_trajs,
            enable_noise=True,
            enable_delay=True,
            discrete_actions=False,
            action_threshold=0.05,
            max_steps=MAX_STEPS_EVALUATION
        )

        # Collect agent trajectories
        print("\n4. Collecting agent trajectories...")
        agent_trajs = collect_agent_trajectories(policy, eval_env, n_episodes=30)

        # Compute reproducibility metrics
        print("\n5. Computing reproducibility metrics...")
        expert_states = [positions_to_states(traj) for traj in all_trajs]
        expert_repro = compute_reproducibility_metrics(expert_states)
        agent_repro = compute_reproducibility_metrics(agent_trajs)

        print("\n📊 Reproducibility Comparison:")
        print(f"  Expert combined score: {expert_repro.get('combined_score', np.nan):.4f}")
        print(f"  Agent  combined score: {agent_repro.get('combined_score', np.nan):.4f}")

        # Compute fast comparison metrics
        print("\n6. Computing fast comparison metrics...")
        comparison_metrics = compute_fast_comparison_metrics(expert_states, agent_trajs)
        
        print("📈 Fast Comparison Metrics:")
        for key, value in comparison_metrics.items():
            print(f"  {key}: {value:.4f}")

        # ====================== 7. Save trajectories to CSV ======================
        print("\n7. Saving trajectories to CSV files...")
        
        trajectory_dir = "trajectory_data_fast"
        Path(trajectory_dir).mkdir(exist_ok=True)
        
        # Save expert trajectories
        expert_csv_path = os.path.join(trajectory_dir, "expert_trajectories.csv")
        save_expert_trajectories_csv(all_trajs, labels, expert_csv_path)
        
        # Save agent trajectories  
        agent_csv_path = os.path.join(trajectory_dir, "agent_trajectories.csv")
        save_agent_trajectories_csv(agent_trajs, agent_csv_path)
        
        # Save comparison file
        comparison_csv_path = os.path.join(trajectory_dir, "comparison_trajectories.csv")
        save_comparison_trajectories_csv(all_trajs, labels, agent_trajs, comparison_csv_path)
        
        print(f"All trajectories saved to {trajectory_dir}/ directory")

        # ====================== 8. Visualization ======================
        print("\n8. Generating visualizations...")
        
        # Plot trajectory comparisons
        plot_trajectory_comparison(all_trajs[:3], agent_trajs[:3])
        
        # Plot survival analysis
        plot_survival_analysis(agent_trajs)
        
        # ====================== 9. Summary Report ======================
        print("\n9. Generating summary report...")
        
        summary = {
            'expert_trajectories': len(all_trajs),
            'agent_trajectories': len(agent_trajs),
            'expert_reproducibility': expert_repro,
            'agent_reproducibility': agent_repro,
            'comparison_metrics': comparison_metrics,
            'mean_survival_steps': float(np.mean([len(traj) for traj in agent_trajs])) if agent_trajs else 0.0,
            'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        analysis_dir = "analysis_results_fast"
        Path(analysis_dir).mkdir(exist_ok=True)
        with open(os.path.join(analysis_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=_json_serial)
        
        print("\n" + "="*60)
        print("FAST EVALUATION COMPLETE - SUMMARY")
        print("="*60)
        print(f"Expert trajectories analyzed: {len(all_trajs)}")
        print(f"Agent trajectories generated: {len(agent_trajs)}")
        print(f"Mean survival steps: {summary['mean_survival_steps']:.1f}")
        print(f"Expert reproducibility: {expert_repro.get('combined_score', 0):.4f}")
        print(f"Agent reproducibility: {agent_repro.get('combined_score', 0):.4f}")
        print(f"Length ratio (agent/expert): {comparison_metrics.get('length_ratio', 0):.4f}")
        print("\nResults saved to:")
        print(f"  - {trajectory_dir}/ (trajectory CSV files)")
        print(f"  - {analysis_dir}/ (analysis results and visualizations)")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please ensure the dataset file and model checkpoints exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_evaluation()
