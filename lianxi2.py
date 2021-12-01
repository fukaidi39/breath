# Author: Mr.Wang
# CreateTime: 2021/11/11
# FileName: lianxi2
import test
import threading
import math
import time
import numpy as np
import cv2
from queue import Queue
from threading import Thread
# A thread that produces data
d = 0
def producer():
    global d
    while True:
        d += 1








# Create the shared queue and launch both threads