#!/bin/sh
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install requests pandas psycopg2-binary websocket-client websockets statsmodels
