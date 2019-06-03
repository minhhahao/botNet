from django.conf import settings
from django.apps import AppConfig
import logging
import sys
import os
from ..bot import botNet

chatbotPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(chatbotPath)

logger = logging.getLogger(__name__)


class botManager(AppConfig):
    '''
    Manage a single instance of the chatbot shared over the website
    '''
    name = 'interface'
    verbose_name = 'Interface'

    bot = None

    def ready(self):
        '''
        Called by Django only once during startup
        '''
        # Initialize the chatbot daemon (should be launched only once)
        # HACK: Avoid the autoreloader executing the startup code twice (could also use: python manage.py runserver --noreload)
        # (see http://stackoverflow.com/questions/28489863/why-is-run-called-twice-in-the-django-dev-server)
        if (os.environ.get('RUN_MAIN') == 'true' and not any(x in sys.argv for x in ['makemigrations', 'migrate'])):  # HACK: Avoid initialisation while migrate
            botManager.init_bot()

    @staticmethod
    def init_bot():
        ''' Instantiate the chatbot for later use
        Should be called only once
        '''
        if not botManager.bot:
            logger.info('Initializing bot...')
            botManager.bot = botNet.botNet()
            botManager.bot.main(['--model_tag', 'drunkboiv1', '--mode', 'daemon', '--root_dir', chatbotPath])
        else:
            logger.info('Bot already initialized.')

    @staticmethod
    def call_bot(sentence):
        '''
        Use the previously instantiated bot to predict a response to the given sentence
        Args:
            sentence <str>: the question to answer
        Return:
            str: the answer
        '''
        if botManager.bot:
            return botManager.bot.daemon_predict(sentence)
        else:
            logger.error('Error: Bot not initialized!')
