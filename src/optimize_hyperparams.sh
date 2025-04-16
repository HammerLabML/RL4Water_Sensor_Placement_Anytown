#!/bin/zsh

python train.py --algo dqn --env AnytownDiscrete-v0 --n-timesteps 240000 --optimization-log-path ../Results/No_Uncertainty/Anytown/Hyperparameter_Optimization/Trial_58-137 --device cpu --n-trials 80 --optimize-hyperparameters --n-jobs 1 --sampler tpe --pruner halving --n-evaluations 2 --study-name "anytown1-60"
