#!/bin/zsh

python sequential_pump_speed_finder.py \
../Data/Anytown/ATM.inp \
../Data/Anytown/ATM_constraints.yaml \
../Data/Parameters_for_Optimization/objective_weights_1.yaml \
../Data/Parameters_for_Optimization/initial_optimizer_params_3.yaml \
--output_file ../Results/Initial_Pump_Speeds/Anytown/output_8.yaml \
--only_initial
