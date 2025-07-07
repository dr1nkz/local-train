#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate
pip install super-gradients
pip install torchinfo