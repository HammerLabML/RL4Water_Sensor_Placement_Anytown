from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2 import cluster_work
import yaml

class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        print(config['params'])
        print('-'*20)
        print()

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass

if __name__ == "__main__":
    cw = cluster_work.ClusterWork(MyExperiment)
    cw.run()

