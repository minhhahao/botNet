from channels import Group
from channels.sessions import channel_session
import logging
import sys
import json

from .botmanager import botManager


logger = logging.getLogger(__name__)


def _get_client_name(client):
    '''
    Return the unique id for the client
    Args:
        client list<>: the client which send the message of the from [ip (str), port (int)]
    Return:
        str: the id associated with the client
    '''
    return 'room-' + client[0] + '-' + str(client[1])


@channel_session
def ws_connect(message):
    '''
    Called when a client try to open a WebSocket
    Args:
        message <obj>: object containing the client query
    '''
    if message['path'] == '/chat':  # Check we are on the right channel
        client_name = _get_client_name(message['client'])
        logger.info('New client connected: {}'.format(client_name))
        Group(client_name).add(message.reply_channel)  # Answer back to the client
        message.channel_session['room'] = client_name
        message.reply_channel.send({'accept': True})


@channel_session
def ws_receive(message):
    '''
    Called when a client send a message
    Args:
        message <obj>: object containing the client query
    '''
    # Get client info
    client_name = message.channel_session['room']
    data = json.loads(message['text'])

    # Compute the prediction
    question = data['message']
    try:
        answer = botManager.call_bot(question)
    except:  # Catching all possible mistakes
        logger.error('{}: Error with this question {}'.format(client_name, question))
        logger.error("Unexpected error:", sys.exc_info()[0])
        answer = 'Error: Internal problem'

    # Check eventual error
    if not answer:
        answer = 'Error: Try a shorter sentence'

    logger.info('{}: {} -> {}'.format(client_name, question, answer))

    # Send the prediction back
    Group(client_name).send({'text': json.dumps({'message': answer})})

@channel_session
def ws_disconnect(message):
    '''
    Called when a client disconnect
    Args:
        message <obj>: object containing the client query
    '''
    client_name = message.channel_session['room']
    logger.info('Client disconnected: {}'.format(client_name))
    Group(client_name).discard(message.reply_channel)
