# ClearML Agent Demo

This repository demonstrates how to set up and run [ClearML](https://clear.ml/) agents using [Podman](https://podman.io/) on both local and remote machines.


## Remote Worker Setup

### 1. Install and Configure Podman

Ensure Podman is installed and set up correctly. Then configure the NVIDIA Container Toolkit (for GPU support):

* [Installing the NVIDIA Container Toolkit](httphttps://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
* [Generating a CDI specification](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html#generating-a-cdi-specification)

### 2. Clone the Repository

```bash
git clone https://github.com/jeremiedecock/clearml-agent-demo.git
cd clearml-agent-demo
```

### 3. Add Your ClearML Configuration

Copy your `clearml.conf` file into the cloned directory. This file contains your ClearML server credentials. **Do not commit it to version control.**

### 4. Create the ClearML Agent Podman Image

```bash
./build-clearml-agent.sh
```

### 5. Create Queues on the ClearML Server

Ensure the following queues exist on your ClearML server (e.g., [https://app.clear.ml/workers-and-queues/queues](https://app.clear.ml/workers-and-queues/queues)):

* `worker-cpu`
* `worker-bi-gpu`
* `worker-single-gpu`

### 6. Launch ClearML Agent Containers

Start ClearML agents in separate Podman containers. A `tmux` session is recommended to manage multiple terminals:

```bash
./run-clearml-agent-worker-cpu.sh
./run-clearml-agent-worker-bi-gpu.sh
./run-clearml-agent-worker-gpu0.sh
./run-clearml-agent-worker-gpu1.sh
```


## Local Development Setup

### 1. Install and Configure Podman

Ensure Podman is installed and working on your local machine.

### 2. Clone the Repository

```bash
git clone https://github.com/jeremiedecock/clearml-agent-demo.git
cd clearml-agent-demo
```

### 3. Add Your ClearML Configuration

Copy your `clearml.conf` file into the cloned directory (never commit it).

### 4. Build the Local Dev Container

```bash
./build.sh
```


## Running Experiments

### Local Execution

Run a script locally inside the development container:

```bash
./run.sh getting_started_multiclass_classification_mnist_dense_layer_clearml.py
```

### Remote Execution

Submit the script to a remote agent queue (e.g., `worker-bi-gpu`):

```bash
./run.sh getting_started_multiclass_classification_mnist_dense_layer_clearml.py --remote --remote-queue worker-bi-gpu
```

### Hyperparameter Optimization (HPO)

Run HPO with ClearML (requires a valid `task_id`):

```bash
./run.sh getting_started_run_hpo_service_locally_but_workers_remotely.py --task-id <task_id>
```

your can find the `task_id` in the ClearML web UI after submitting a task.


## Notes

* All scripts assume Podman is properly configured (including GPU access).
* Make sure your ClearML server is reachable from both local and remote machines.
* For security: Never commit sensitive files like `clearml.conf` to version control.
