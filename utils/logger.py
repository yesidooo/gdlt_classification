import logging


def test_log():
    logging.basicConfig(level=logging.DEBUG, filename='log.txt', format='%(asctime)s - [%(filename)s-->line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger('root')
    format = logging.Formatter('%(asctime)s - [%(filename)s-->line:%(lineno)d] - %(levelname)s: %(message)s')

    handle = logging.StreamHandler()
    handle.setLevel(level=logging.INFO)
    handle.setFormatter(fmt=format)
    logger.addHandler(handle)

    logger.debug('1')
    logger.info('2')
    logger.error('4')


if __name__ == '__main__':
    test_log()