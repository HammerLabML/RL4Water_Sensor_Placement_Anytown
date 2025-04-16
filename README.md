# To Pump or Not to Pump - Sensor-based Reinforcement Learning for an Optimal Scheduler

Official Code for the workshop paper "To Pump or Not to Pump - Sensor-based Reinforcement
Learning for an Optimal Scheduler" (under submission).
All system and package requirements are listed in the document 'rl4water.yml'. 
A corresponding conda environment can be setup via `conda env create -f rl4water.yml`.

## Execution of Experiments

To manage our experiments and make them easily executable on slurm clusters we use the ClusterWorks2 Package.
The experiment configurations are specified in .yml files.
Each experiment has a name. To execute one of the experiments you have to pass the yml File and the experiment name to the program.

```
cd src/
python cw_main.py Configs/Anytown/anytown.yml -e EXPERIMENT_NAME 
```

The location for the results to be saved in is also specified in the .yml File.

## Citation

### Repository:
```
@misc{Rl4Water,
  author        = {Alissa Müller and Paul Stahlhofen and Hammer, Barbara},
  title         = {{To Pump or Not to Pump - Sensor-based Reinforcement
Learning for an Optimal Scheduler}},
  year          = {2025},
  publisher     = {GitHub}
  journal       = {GitHub repository},
  organization  = {CITEC, Bielefeld University, Germany},
  howpublished  = {\url{https://github.com/HammerLabML/RL4Water_Sensor_Placement_Anytown}},
}
```


## Acknowledgments
We gratefully acknowledge funding from the European
Research Council (ERC) under the ERC Synergy Grant Water-Futures (Grant
agreement No. 951424). 