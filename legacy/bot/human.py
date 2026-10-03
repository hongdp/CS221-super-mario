from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from agent import *
from time import sleep
import pickle


class HumanAgent(Agent):
    def __init__(self, options):
        self.action = [0, 0, 0, 0, 0, 0]
        self.state = None
        # with keyboard.Listener(
        #    on_press=self.on_press,
        #    on_release=self.on_release) as listener:
        #    self.listener = listener

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)
        self.log(self.action, reward)
        # print('acting {}'.format(self.action))
        # sleep(1.0/30)
        sleep(0.1)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        (obs, reward, is_finished, info) = self.state
        return is_finished

    def handle(self, e):
        # self.listener.join()
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

        # def on_press(self, key):
        #    if key.char == 'a':
        #        self.action[1] = 1
        #    elif key.char == 'd':
        #        self.action[3] = 1
        #    elif key.char == 'j':
        #        self.action[4] = 1
        #    elif key.char == 'k':
        #        self.action[5] = 1

        # def on_release(self, key):
        #    if key.char == 'a':
        #        self.action[1] = 0
        #    elif key.char == 'd':
        #        self.action[3] = 0
        #    elif key.char == 'j':
        #        self.action[4] = 0
        #    elif key.char == 'k':
        #        self.action[5] = 0
