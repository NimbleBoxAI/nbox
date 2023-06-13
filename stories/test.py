from time import sleep
from nbox import logger, lo

def main(x: int = 4):
  logger.info(lo("Hello World!", x = x))
  sleep(5)
