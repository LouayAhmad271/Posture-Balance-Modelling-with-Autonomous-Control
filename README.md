# Human Balance Modelling with GAIL and Surrogate-Assisted Reward Design

This repository contains code for learning human-like standing balance using **Generative Adversarial Imitation Learning (GAIL)** and a **surrogate-assisted reward optimization** pipeline. The project includes:

- A custom `HumanBalanceEnv` (Gymnasium) with realistic biomechanics (inverted pendulum model, sensorimotor noise, delays).
- GAIL training with PPO, progressive episode length, survival‑based early stopping.
- Enhanced trajectory comparison metrics (spectral analysis, PCA, t‑SNE).
- A separate surrogate‑assisted reward design for rhythmic standing balance.

## Repository Structure

── README.md

├── LICENSE
├── requirements.txt
├── run.py # Main entry point (training + evaluation)
├── Enhanced_Agent.py # GAIL training with enhanced metrics
├── RL.py # Alternative GAIL implementation (similar to Enhanced_Agent)
├── Eval.py # Fast evaluation script for trained models
├── heatmaps.ipynb # Jupyter notebook for PD controller simulations
├── surrogate_assisted_reward_design_for_learning_rhythmic_standing_balance.py
└── checkpoints_stable/ # Created during training; stores best models
