# CLAUDE.md


## Project
PyTorch training network for the reversi boardgame. Python 3.12, Pytorch 2.x, Numpy 1.2.x. The project creates a training network for the reversi boardgame. The training pipeline can be found in the /training folder. 

## Important Files
training/TRAINING_README.md - the outline of the training architecture
training/evaluate.py - the base evaulation code
training/generate_data.py - the data generation code
training/train_nn.py - the training network code
training/run_pipeline.py - the script that runs the training pipeline

## Training Architecture
The training architecture can be found in /training/TRAINING_README.md and /training/ARCHITECTURE.md
- src/weights - the candidate weights and the baseline weights
- training/data - the generate data
- src/ contains the different hueristic functions and scripts to play the game outside of training for human evaluation
- Total number of NN nodes needs to stay under 1 million

## Conventions
- Use pytorch-patterns skill for all model/training code
- Device-agnostic code (no hardcoded `.cuda()`)
- All experiments must be reproducible (seeded)
