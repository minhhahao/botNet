:: bat file for Windows users

@ECHO OFF
cd chat\
redis-server &
python manage.py runserver %*
pause
