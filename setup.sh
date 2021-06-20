#!/usr/bin/env bash
virtualenv -p $(which python3) .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download stl files
wget "https://minersutep-my.sharepoint.com/:u:/g/personal/mhassan_miners_utep_edu/EVh0eFTUsgdItUAPHVwrJ3wBWVIaCpircgqvhevYA98JWQ?e=PAn8vn&download=1" -O stls.tar.gz
tar -xzvf stls.tar.gz && rm -f stls.tar.gz

