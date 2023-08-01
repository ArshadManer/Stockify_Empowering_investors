
from stockify.pipeline.pipeline import Pipepline

from stockify.logger import logging


def main():
    try:
        pipepline = Pipepline()
        pipepline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
        main()