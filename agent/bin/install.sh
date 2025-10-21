#!/usr/bin/env bash
set -e

read -p "First Install? (Y/N): " answer

if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
    echo "Running first-time setup..."
    python3 -m venv venv
else
    echo "Update/Reinstallation"
fi

source venv/bin/activate
pip install -r agent/requirements.txt
cd agent/bin/
python setup.py install