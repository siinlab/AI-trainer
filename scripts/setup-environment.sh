#!/bin/bash
set -e 

cd "$(dirname "$0")/.."

# Ensure this is an ubuntu system
if [ ! -f /etc/lsb-release ]; then
    echo "This script is intended for Ubuntu systems only."
    exit 1
fi

# Update package list
export DEBIAN_FRONTEND=noninteractive
sudo apt update -y

# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.11
uv python install 3.11

# Create a virtual environment
uv venv --python 3.11

# Install required packages
uv sync

# Enable the virtual environment
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Enable auto-completion for click
echo 'eval "$(_SIIN_TRAINER_COMPLETE=bash_source siin-trainer)"' >> ~/.bashrc

# Enable WandB
yolo settings wandb=True
