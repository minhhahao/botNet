from . import views
from django.conf.urls import url

from .botmanager import botManager

urlpatterns = [
    url(r'^$', views.main_view),
]
