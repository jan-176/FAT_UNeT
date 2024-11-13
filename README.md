<img src="https://upload.wikimedia.org/wikipedia/commons/3/3b/Hochschule_Amberg-Weiden_Logo_2013.svg" alt="Image 1" width="150"/>

# Master's Thesis 

This repository supplements my master's thesis "Unsupervised Anomaly Detection in Multivariate Time Series Data using Deep Learning". This is a refactored version of the code used to obtain the results in the master's thesis for ease of use. The code is provided as-is and serves as starting point for further research. Due to limited resources, I am unable to provide support on any issues you may experience with installing or running the tool.

## Installation
The code was executed using Python-3.10. Run the following command to install all necessary python libraries:

```bash
pip3 install -r requirements.txt
```

## Experiments

To reproduce an experiment, run the following command:
```bash
python code.py <experiment> --config <config> 
```

where `<experiment>` can be `experiment1_2` or `experiment3`, and the respective configuration files are at the folder `config` located.

The configuration file to run `experiment 1` for the `SMD` dataset utilizing `FAT` looks like: 
```json
{
    "model_name": "FAT",
    "dataset": "SMD",
    "entity": "machine-1-1",
    "retrain": "True",
    "shuffle": "False",
    "val": 0,
    "mlflow_experiment": "Experiment_1",
    "n_epochs": 5,
    "hyp_lr": 0.0001,
    "hyp_criterion": "MSE",
    "hyp_percentage": 1.0,
    "hyp_lm_d0": 0.99995,
    "hyp_lm_d1": 1.06,
    "hyp_delta": 1.0
}
```
and the command to perform the above displayed experiment looks like: 

```bash
python code.py experiment1_2 --config config/exp1/smd.json 
```

The results are saved in mlflow. To see the results run the following command: 
```bash
mlflow ui
```
The results of the respective experiment are illustrated and reachable at `localhost:5000`.

## License

BSD-3-Clause.

Copyright (c) 2022, Jan Schuster, Lingrui Yu, Shreshth Tuli.
All rights reserved.

See License file for more details.

