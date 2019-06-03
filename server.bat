:: bat file for Windows users

@ECHO OFF
ECHO Running server in the background. Locate to localhost:8000 for interactive session
cd botsite\
redis-server &
python manage.py runserver %*
pause
