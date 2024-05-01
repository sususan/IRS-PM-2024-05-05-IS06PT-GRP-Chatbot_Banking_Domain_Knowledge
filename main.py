from configurations import logger


def main():
    logsubhead = 'main.py.__init__()-'
    logger.info(f"{logsubhead} start")
    from bot import main as start
    start()


def __init__():
    main()


if __name__ == '__main__':
    main()
