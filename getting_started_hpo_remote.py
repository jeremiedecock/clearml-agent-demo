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
    parser.add_argument('--task-id', type=int, help='Task to optimize')
    return parser.parse_args()


# SETUP LOGGING ###############################################################

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)


# MAIN EXECUTION ###############################################################

def train_model(args: argparse.Namespace) -> int:
    """
    Train the neural network model on the MNIST dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing training configuration.
    """

    task: Optional[Task] = None
    try:
        # Always initialize ClearML before anything else to let automatic hooks track as much as possible
        task = Task.init(
            project_name="Snippets",
            task_name="MNIST Dense Layers",
            task_type=Task.TaskTypes.optimizer, # TODO?
            reuse_last_task_id=False,           # TODO?
        )

        # Set the Git repository for the agent to use a read-only public repository that does not require authentication
        # task.set_script(repository='https://github.com/jeremiedecock/clearml-agent-demo.git')
        task.set_repo(repo='https://github.com/jeremiedecock/clearml-agent-demo.git')

        logger.info(f"ClearML task {task.id} initialized successfully")

        if args.remote:
            logger.info("Executing task remotely, exiting process")
            task.execute_remotely(
                queue_name=args.remote_queue,
                clone=False,
                exit_process=True,              # TODO ???
            )
    except Exception as e:
        logger.warning(f"Failed to initialize ClearML task: {e}")


def hyper_parameter_optimization(clearml_task_id: int):
    logger.info(f"Optimize hyper parameters of task {clearml_task_id}...")

    # INITIALIZING CLEARML HPO TASK ###########################################

    # TODO ?
    # task: Optional[Task] = None
    try:
        # TODO ?
        # task = Task.init(
        #     project_name="Snippets",
        #     task_name="MNIST Dense Layers HPO",
        #     task_type=Task.TaskTypes.optimizer,
        #     reuse_last_task_id=False
        # )

        # Set the Git repository for the agent to use a read-only public repository that does not require authentication
        # task.set_script(repository='https://github.com/jeremiedecock/clearml-agent-demo.git')
        # TODO ?
        # task.set_repo(repo='https://github.com/jeremiedecock/clearml-agent-demo.git')

        hp_optimizer = HyperParameterOptimizer(
            # specifying the task to be optimized, task must be in system already so it can be cloned
            base_task_id=TEMPLATE_TASK_ID,
            # setting the hyperparameters to optimize
            hyper_parameters=[
                # UniformIntegerParameterRange('epochs', min_value=2, max_value=24, step_size=2),
                # UniformIntegerParameterRange('batch_size', min_value=32, max_value=96, step_size=16),
                UniformParameterRange('dropout_rate', min_value=0, max_value=0.5, step_size=0.05),
                # UniformParameterRange('lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
            ],
            # setting the objective metric we want to maximize/minimize
            objective_metric_title='Accuracy',
            objective_metric_series='test',    # TODO?
            objective_metric_sign='max',
            # setting hp_optimizer
            optimizer_class=OptimizerOptuna,
            # configuring optimization parameters
            execution_queue=args.remote_queue,
            max_number_of_concurrent_tasks=2,
            optimization_time_limit=60.,
            compute_time_limit=120,
            total_max_jobs=20,
            min_iteration_per_job=15000,
            max_iteration_per_job=150000,
        )

        if args.remote:
            logger.info("Executing HPO task remotely, exiting process")
            hp_optimizer.start()
        else:
            logger.info("Executing HPO task locally")
            hp_optimizer.start_locally()

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
