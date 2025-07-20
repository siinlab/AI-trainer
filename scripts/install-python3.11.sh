#!/bin/bash

# Ensure this is an ubuntu system
if [ ! -f /etc/lsb-release ]; then
    echo "This script is intended for Ubuntu systems only."
    exit 1
fi

# Update package list
export DEBIAN_FRONTEND=noninteractive
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# Install pip for Python 3.11
curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.11 - --ignore-installed

# Create a symlink for python3
sudo ln -fs "$(which python3.11)" "$(which python3)"
sudo ln -fs "$(which python3.11)" "$(which python)"
# Create a symlink for pip
sudo ln -fs "$(which pip3.11)" "$(which pip)"
sudo ln -fs "$(which pip3.11)" "$(which pip3)"