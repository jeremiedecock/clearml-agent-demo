#!/usr/bin/env python3
# coding: utf-8

# Documentation:
# - https://clear.ml/docs/latest/docs/guides/optimization/hyper-parameter-optimization/examples_hyperparam_opt/
# - https://github.com/clearml/clearml/blob/master/examples/optimization/hyper-parameter-optimization/hyper_parameter_optimizer.py
# - https://clear.ml/docs/latest/docs/references/sdk/hpo_optimization_hyperparameteroptimizer/

import argparse
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna
from clearml import Task
import logging
from typing import Optional


# CONFIGURATION ###############################################################

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for MNIST classification.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='ClearML HPO Snippet')
    parser.add_argument('--task-id', type=str, help='Task to optimize')
    return parser.parse_args()


# SETUP LOGGING ###############################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)


# MAIN EXECUTION ###############################################################

def hyper_parameter_optimization(clearml_task_id: str):
    logger.info(f"Optimize hyper parameters of task {clearml_task_id}...")

    # INITIALIZING CLEARML HPO TASK ###########################################

    task: Optional[Task] = None
    try:
        task = Task.init(
            project_name="Snippets",
            task_name="HPO",
            task_type=Task.TaskTypes.optimizer,
            reuse_last_task_id=False
        )

        # Set the Git repository for the agent to use a read-only public repository that does not require authentication
        # task.set_script(repository='https://github.com/jeremiedecock/clearml-agent-demo.git')
        task.set_repo(repo='https://github.com/jeremiedecock/clearml-agent-demo.git')

        hp_optimizer = HyperParameterOptimizer(
            # specifying the task to be optimized, task must be in system already so it can be cloned
            base_task_id=clearml_task_id,
            # setting the hyperparameters to optimize
            hyper_parameters=[
                # UniformIntegerParameterRange('Args/epochs', min_value=2, max_value=24, step_size=2),
                # UniformIntegerParameterRange('Args/batch_size', min_value=32, max_value=96, step_size=16),
                # UniformParameterRange('Args/dropout_rate', min_value=0, max_value=0.5, step_size=0.05),
                UniformParameterRange('Args/lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
                UniformIntegerParameterRange('Args/num_hidden_layers', min_value=1, max_value=4, step_size=1),
                UniformIntegerParameterRange('Args/hidden_layer_size', min_value=16, max_value=512, step_size=16),
            ],
            # setting the objective metric we want to maximize/minimize
            objective_metric_title='Accuracy',
            objective_metric_series='test',      # TODO
            objective_metric_sign='max',
            # setting hp_optimizer
            optimizer_class=OptimizerOptuna,
            # configuring optimization parameters
            execution_queue="worker-single-gpu",     # TODO: La queue d'execution des taches executées (et non pas du service HPO)
            max_number_of_concurrent_tasks=8,
            optimization_time_limit=60.,
            compute_time_limit=120,
            total_max_jobs=50,
            min_iteration_per_job=15000,
            max_iteration_per_job=150000,
        )

        logger.info(f"Executing HPO task {task.id} remotely, exiting process")
        task.execute_remotely(queue_name='hpo-coordinator', exit_process=True)    # TODO: <<<

        hp_optimizer.start()

        # logger.info("Executing HPO task locally")
        # hp_optimizer.start_locally()

        # hp_optimizer.set_time_limit(in_minutes=120.0)
        hp_optimizer.wait()
        top_hp = hp_optimizer.get_top_experiments(top_k=3)
        print([t.id for t in top_hp])
        hp_optimizer.stop()

    except Exception as e:
        logger.warning(f"Failed to initialize ClearML task: {e}")


if __name__ == "__main__":
    """
    Main entry point for training and testing the MNIST classifier.
    """
    try:
        # Parse command line arguments
        args: argparse.Namespace = parse_args()

        # Test the model
        hyper_parameter_optimization(args.task_id)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
