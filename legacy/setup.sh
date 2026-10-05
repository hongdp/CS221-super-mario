#!/bin/bash
sudo apt-get install fceux

virtualenv .env
source .env/bin/activate
python setup.py
pip install -r requirements.txt
