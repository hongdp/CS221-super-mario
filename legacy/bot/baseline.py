import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *

class BaselineAgent(Agent):
    def __init__(self, options):
        self.action = [0,0,0,0,0,0]
        self.state = None

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)

        ahead = get_tile_from_mario(obs, 2, 0)
        if ahead is not None:
            print('ahead ' + Tile.name(ahead))
            if ahead == Tile.EMPTY_SPACE:
                self.action = Action.act('Right')
            elif ahead in [Tile.ENEMY, Tile.OBJECT]:
                self.action = Action.act('A')
        self.log(self.action, reward)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        (obs, reward, is_finished, info) = self.state
        total_score = info["distance"]
        stop = total_score > 32000
        if is_finished or stop:
            if stop:
                print('exiting the game cuz I no longer wanna play ...')
            return True
        else:
            return False

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

