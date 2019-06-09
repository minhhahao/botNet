from bot import botNet
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    bot = botNet.botNet()
    bot.main()
