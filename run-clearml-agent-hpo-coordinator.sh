#!/bin/sh

# To use Nvidia GPUs with Podman, a CDI specification of the installed device(s) have to be made first:
# 1. Generate the CDI specification file: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
# 2. (Optional) Check the names of the generated devices: `nvidia-ctk cdi list`
#
# More info: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html#procedure

# TODO: this ClearML Agent should be run in "service" mode but this requires to use the "docker" mode too (https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_services_mode/)

podman run --rm -it \
           --name="clearml-agent-hpo-coordinator" \
           -e CLEARML_AGENT_PACKAGE_PYTORCH_RESOLVE=none \
           -e CLEARML_WORKER_ID="$(hostname):hpo-coordinator" \
           -v clearml-agent-cache-hpo-coordinator:/root/.clearml \
           localhost/clearml-agent:latest clearml-agent daemon --queue "hpo-coordinator" --foreground
