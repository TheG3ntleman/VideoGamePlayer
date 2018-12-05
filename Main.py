import cv2
import numpy as np
from PIL import ImageGrab
from pynput import keyboard
from pynput.keyboard import Key, Controller
import tensorflow as tf
from Brainz import predict
import time
import pickle

DEBUG = False
SHOWCASE = True

keyboard_controller = Controller()
MOVES = {0:'w', 1:'a', 2:'s', 3:'d', 4:Key.space, 5:Key.shift_l}

DISX = 0
DISY = 0

WATCH = False
PLAY = True
SAVE_PLAY = True


if not WATCH:
    SHOWCASE = False

def ClickKey(Key, press_time=0.1):
    keyboard_controller.press(Key)
    time.sleep(press_time)
    keyboard_controller.release(Key)

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


def Logic(screen):
    x = predict(np.asarray(cv2.resize(screen, (80, 60))).reshape([1, 80*60]))
    print(x)
    ClickKey(MOVES[x])

game_play = []

def fixer(k):
    try:
        string = str(k)
        m = ""
        for char in string:
            if not char == "'":
                m += char
        return m
    except:
        return k


def on_press(key):
    try: k = key.char
    except: k = key.name
    if key == keyboard.Key.esc: return False
    game_play.append([Screen_Getter()[1], key])

def record():
    lis = keyboard.Listener(on_press=on_press)
    lis.start()
    lis.join()

def play_main_gameplay():
    for frame in game_play:
        cv2.imshow("NeuralView", frame[0])
        if DEBUG:
            print(frame[1])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def dict_perfect(dictionary, search_val):
    for item, val in dictionary.items():
        if str(val) == str(search_val):
            return item

def one_hot_encode_names(control):
    one_hot_controls = []
    for num in range(len(MOVES)):
        key = fixer(control)
        if dict_perfect(MOVES, key) == num:
            one_hot_controls.append(1)
        else:
            one_hot_controls.append(0)

    return one_hot_controls

def process_gameplay(game_play):
    for frame_num in range(len(game_play)):
        control = game_play[frame_num][1]
        game_play[frame_num][1] = (one_hot_encode_names(control))
        game_play[frame_num][0] = cv2.resize(game_play[frame_num][0], (80, 60))
    return game_play

def loop():
    print("STARTED")
    global game_play
    if PLAY:
        timex = time.time()
        screen = None
        while True:

            # Additional debug permission
            if DEBUG:
                timex = debugger(timex)

            # ScreenData, showcase, etc...
            ScreenInfo = Screen_Getter()
            screen = ScreenInfo[1]
            if ScreenInfo[0]:
                break

            Logic(screen)

    elif WATCH:
        record()
        play_main_gameplay()
        if SAVE_PLAY:
            game_play = process_gameplay(game_play)
            file = open('GamePlay.illuminati', 'wb')
            pickle.dump(game_play, file)
            file.close()


loop()
