"""Shared utilities for adaptive training cycle configuration."""

import json
import os

TRAINING_DIR = os.path.dirname(__file__)
CYCLE_CONFIG_PATH = os.path.join(TRAINING_DIR, 'cycle_config.json')
TRAIN_METRICS_PATH = os.path.join(TRAINING_DIR, 'train_metrics.json')

# Default values matching the current hardcoded constants across the pipeline.
DEFAULTS = {
    # generate_data.py defaults
    'nn_ratio': 0.5,
    'ch_ratio': 0.2,
    'cross_ratio': 0.3,
    'epsilon': 0.08,
    'num_games': 10000,
    'random_opening_moves': 4,

    # train_nn.py defaults
    'epochs': 150,
    'lr': 0.001,
    'heuristic_weight_start': 0.9,
    'heuristic_weight_floor': 0.5,

    # weight matrix perturbation (run_pipeline.py)
    'perturbation_strength': 0.0,
}


def load_cycle_config():
    """Load cycle config from JSON, falling back to DEFAULTS for missing keys."""
    config = dict(DEFAULTS)
    if os.path.exists(CYCLE_CONFIG_PATH):
        with open(CYCLE_CONFIG_PATH, 'r') as f:
            overrides = json.load(f)
        config.update(overrides)
    return config


def load_train_metrics():
    """Load training metrics written by train_nn.py after a training run."""
    if os.path.exists(TRAIN_METRICS_PATH):
        with open(TRAIN_METRICS_PATH, 'r') as f:
            return json.load(f)
    return {}
