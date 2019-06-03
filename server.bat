:: bat file for Windows users

@ECHO OFF
cd botsite\
redis-server &
python manage.py runserver %*
pause
