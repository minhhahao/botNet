#!/bin/bash

cd botsite/

# Launch the server
redis-server & python3 manage.py runserver
