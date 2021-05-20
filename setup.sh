#!/usr/bin/env bash
virtualenv -p $(which python3.7) .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download stl files
wget "https://minersutep-my.sharepoint.com/:u:/g/personal/mhassan_miners_utep_edu/EVQaQ975JBhNlet5g12H8-IBkMcDRuA-f85Qe5Yklf9VSA?e=caze5C&download=1" -O stls.tar.gz
tar -xzvf stls.tar.gz && rm -f stls.tar.gz

