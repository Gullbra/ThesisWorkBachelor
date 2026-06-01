#!/usr/bin/env bash

set -euo pipefail
# -e: exit on error
# -u: error on undefined variables
# -o pipefail: fail a pipeline if any command in it fails

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
