#!/usr/bin/env bash
virtualenv -p $(which python3.7) .venv
source .venv/bin/activate
pip install -r requirements.txt
