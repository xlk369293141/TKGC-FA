from nni.experiment import Experiment
experiment = Experiment('local')


search_space = {
    "rank": {"type": "quniform", "value": [100, 1000, 50]},
    "batch_szie": {"type": "quniform", "value": [128, 2056, 128]},
    "learning_rate": {"type": "loguniform", "value": [0.0001, 0.1]},
    "ratio": {"type": "quniform", "value": [0.1, 0.95, 0.05]},
    "reg": {"type": "quniform", "value": [0.01, 0.9, 0.01]}
}

experiment.config.trial_command = 'python learn.py'
experiment.config.trial_code_directory = 'D:\计算机\代码\TKGC\TKGC-Temp\code'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 100
experiment.config.trial_concurrency = 3

experiment.run(8080)