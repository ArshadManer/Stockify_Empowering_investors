import logging
from datetime import datetime

import os

Log_Dir = "proj_logs"

current_time_stamp = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME = f"log_{current_time_stamp}.log"


os.makedirs(Log_Dir,exist_ok=True)
LOG_FILE_PATH = os.path.join(Log_Dir,LOG_FILE_NAME)
logging.basicConfig(filename=LOG_FILE_PATH,

filemode = 'w',
format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',

level= logging.INFO

)