#!/bin/bash

cd chat/

# Launch the server
redis-server & python3 manage.py runserver
