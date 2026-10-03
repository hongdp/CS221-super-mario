from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from enum import *
from util import *
from itertools import chain


class FeatureExtractor:
    __metaclass__ = ABCMeta

    # Agent's initial action
    @abstractmethod
    def featureSize(self): pass

    @abstractmethod
    def extract(self, feature, state): pass


class TileFeatureExtractor(FeatureExtractor):
    def __init__(self, options):
        self.windowSize = options.tileWindowSize

    def featureSize(self):
        return Window.Width, Window.Height, self.windowSize

    def extract(self, feature, state):
        tiles = state.get_last_n_obs(self.windowSize)
        return np.array(np.transpose(tiles, (2, 1, 0)))


class InfoFeatureExtractor(FeatureExtractor):
    def featureSize(self): return 2

    def extract(self, feature, state):
        info = state.get_last_frame().get_info()
        feature['distance'] = info['distance']
        feature['time'] = info['time']


class VelocityFeatureExtractor(FeatureExtractor):
    def featureSize(self): return 1

    def extract(self, feature, state):
        # extract velocity
        dist_delta = state.get_last_frame().get_info()['distance'] - state.get_frames()[0].get_info()['distance']
        v = dist_delta * 1.0 / len(state.get_frames())
        feature['velocity'] = v

class StuckFeatureExtractor(FeatureExtractor):
    def featureSize(self): return 1

    def extract(self, feature, state):
        frames = state.get_frames()
        return get_stuck(frames) * 1

class MarioFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 3

    def extract(self, feature, state):
        # extract x, y coordinate of Mario
        obs = state.get_last_frame().get_obs()
        coord = get_mario_coord(obs)
        if coord is None:
            marioy, mariox = (0, 0)
            feature['has_mario'] = 0
        else:
            marioy, mariox = coord
            feature['has_mario'] = 1
        feature['marioy'] = marioy
        feature['mariox'] = mariox


class FrontFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 4

    def extract(self, feature, state):
        # extract information about object in front of MARIO
        obs = state.get_last_frame().get_obs()
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, i, 0)
            if horizontal_tile is None:
                feature['ahead_{}_height'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                feature['ahead_{}_height'.format(i)] = Window.Height
                for j in range(Window.Height):
                    if obs[hor_y - j, hor_x] == Tile.EMPTY_SPACE:
                        feature['ahead_{}_height'.format(i)] = j
                        break


class BehindFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 4

    def extract(self, feature, state):
        # extract information about object behind MARIO
        obs = state.get_last_frame().get_obs()
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, -i, 0)
            if horizontal_tile is None:
                feature['behind_{}_height'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                feature['behind_{}_height'.format(i)] = Window.Height
                for j in range(Window.Height):
                    if obs[hor_y - j, hor_x] == Tile.EMPTY_SPACE:
                        feature['behind_{}_height'.format(i)] = j
                        break

class EnemyFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 8

    def extract(self, feature, state):
        obs = state.get_last_frame().get_obs()
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, i, 0)
            if horizontal_tile is None:
                feature['front_{}_enemy'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                if obs[hor_y, hor_x] == Tile.ENEMY:
                    feature['front_{}_enemy'.format(i)] = 1
                else:
                    feature['front_{}_enemy'.format(i)] = 0
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, -i, 0)
            if horizontal_tile is None:
                feature['behind_{}_enemy'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                if obs[hor_y, hor_x] == Tile.ENEMY:
                    feature['behind_{}_enemy'.format(i)] = 1
                else:
                    feature['behind_{}_enemy'.format(i)] = 0

class Height5FeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 1

    def extract(self, feature, state):
        obs = state.get_last_frame().get_obs()
        horizontal_tile = get_coord_from_mario(obs, 1, 0)
        if horizontal_tile is None:
            feature['height_5'] = 0
        else:
            hor_y, hor_x = horizontal_tile
            for j in range(5):
                if obs[hor_y-j, hor_x] == Tile.EMPTY_SPACE:
                    feature['height_5'] = 0
                    return
            feature['height_5'] = 1

class PitFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 1

    def extract(self, feature, state):
        # extract whether there is a pit within distance of 3
        feature['pit_ahead'] = 0
        obs = state.get_last_frame().get_obs()
        coord = get_mario_coord(obs)
        if coord is not None:
            marioy, mariox = coord
            for i in range(1, 4):
                if not out_of_frame(mariox + i, 12) and obs[12, mariox + i] == Tile.EMPTY_SPACE:
                    feature['pit_ahead'] = 1
                    break


class PrevActionsFeatureExtractor(FeatureExtractor):
    def __init__(self, options, actions):
        self.options = options
        self.actions = actions

    def featureSize(self):
        return len(Action.NAME) * self.options.stepCounterMax * self.options.prevActionsSize

    def extract(self, feature, state):
        prevActions = state.get_prev_actions()

        numPrevActions = self.options.prevActionsSize
        prevActions = [0] * (numPrevActions - len(prevActions)) + prevActions

        i = numPrevActions  * self.options.stepCounterMax
        for actions in [self.actions[actionIdx].get_actions() for actionIdx in prevActions]:
            actions = actions + [[Action.NO_ACTION]] * (self.options.stepCounterMax - len(actions))
            for actionList in [Action.act(a) for a in actions]:
                for j, a in enumerate(actionList):
                    feature['prevActions-{}-Bit{}'.format(i, j)] = a 
                i -= 1

class PrevActionIndexFeatureExtractor(FeatureExtractor):
    def __init__(self, options, actions):
        self.options = options
        self.actions = actions

    def featureSize(self):
        return len(self.actions) * self.options.prevActionsSize

    def extract(self, feature, state):
        prevActions = state.get_prev_actions()

        numPrevActions = self.options.prevActionsSize
        prevActions = [0] * (numPrevActions - len(prevActions)) + prevActions

        for i, actionIdx in enumerate(prevActions):
            for j in range(len(self.actions)):
                feature['prevActions-{}-Index{}'.format(len(prevActions) - i, j)] = (actionIdx == j) * 1

class PrevActionAFeatureExtractor(FeatureExtractor):
    def __init__(self, options, actions):
        self.options = options
        self.actions = actions

    def featureSize(self):
        return 1 

    def extract(self, feature, state):
        actionIdxs = state.get_prev_actions()
        if len(actionIdxs) < 1:
            prevA = 0
        else:
            actions = self.actions[actionIdxs[-1]].get_actions()
            if len(actions) < self.options.stepCounterMax:
                prevA = 0
            else:
                prevA = actions[-1][Action.index('A')]
        feature['prevActionA'] = prevA
