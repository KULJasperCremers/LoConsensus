import logging

from locomotif.loconsensus import timeseries_generator as tsg

logging.basicConfig(
    level=logging.INFO,
    # level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

if __name__ == '__main__':
    LOGGER = logging.getLogger(__name__)
    tsg.test()
