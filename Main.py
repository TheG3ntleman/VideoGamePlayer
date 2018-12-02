import cv2
import numpy as np
from PIL import ImageGrab
from Keys import *
import random
import time

DEBUG = True
SHOWCASE = True

MOVES = [W, A, S, D, SPACE]
DISX = 0
DISY = 0

WATCH = False
PLAY = True

if not PLAY:
    SHOWCASE = False

def process_frame(img, theta1=220, theta2=300):
    return cv2.GaussianBlur(cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), theta1, theta2), (5, 5), 0)


def debugger(timex):
    print(time.time() - timex)
    return time.time()


def Screen_Getter():
    # Obtaining screen and blitting
    screen = process_frame(np.asarray(ImageGrab.grab((10, 38, 970, 575))))
    g = False
    if SHOWCASE:
        cv2.imshow('NeuralView', screen)

        # Additional Clockworks
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            g = True

    return g, screen


def Logic():
    rand = random.randrange(0, 4)
    ClickKey(MOVES[rand])

def record(lenght):
    start_time = time.time()
    while lenght:
        pass


def loop():
    if PLAY:
        timex = time.time()
        screen = None
        while True:

            # Additional debug permission
            if DEBUG:
                timex = debugger(timex)

            Logic()

            # ScreenData, showcase, etc...
            ScreenInfo = Screen_Getter()
            screen = ScreenInfo[1]
            if ScreenInfo[0]:
                break
    elif WATCH:
        record()


time.sleep(1)
loop()
