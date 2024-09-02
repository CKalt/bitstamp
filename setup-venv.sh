#!/bin/bash

   # Create a new virtual environment
   python3 -m venv env

   # Activate the virtual environment
   source env/bin/activate

   # Upgrade pip
   pip install --upgrade pip

   # Install compatible versions of required packages
   pip install websockets==9.1
   pip install requests==2.27.1
   pip install urllib3==1.26.15
   pip install pandas==1.3.5
   pip install psycopg2-binary==2.9.3
   pip install websocket-client==1.3.3
   pip install statsmodels==0.13.2
   pip install httpx==0.23.0
   pip install matplotlib==3.5.2  # Add this line

   # Optional: Install any other required packages
   # pip install package-name==version

   # Print installed packages
   pip list

   echo "Virtual environment setup complete!"
