#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install super-gradients
pip install torchinfo
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117