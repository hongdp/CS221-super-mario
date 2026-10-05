from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from enum import *


# Agent interface to interact with emulator
class Agent:
    __metaclass__ = ABCMeta

    # Agent's initial action 
    @abstractmethod
    def initAction(self): pass

    # Agent returns a action based on the feedback from emulator
    @abstractmethod
    def act(self, obs, reward, is_finished, info): pass

        # obs: 13 x 16 numpy array (y, x). (0, 0) is the top left corner

        # info dict
        # A value of -1 indicates that the value is unknown
        # distance = info['distance'] # Total distance from the start (x-axis)
        # level = info['level']
        # coins = info['coins'] # The current number of coins
        # player_status = info['player_status'] # Indicates if Mario is small (value of 0), big (value of 1), or can shoot fireballs (2+)
        # score = info['score'] # The current score
        # time = info['time'] # # The current time left
        # ignore = info['ignore'] # Will be added with a value of True if the game is stuck and is terminated early


    # Agent can determine when does it want to stop playing the game
    @abstractmethod
    def exit(self): pass

    # Error handling 
    @abstractmethod
    def handle(self, e): pass

    def log(self, action, reward):
        print('acting {} reward={}'.format(','.join(Action.names(action)), reward))
