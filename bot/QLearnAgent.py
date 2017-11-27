import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
from QLearnAlgo import *


class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = Action.act('Right')
        self.frame = None
        self.maxGameIter = options.maxGameIter
        self.windowsize = options.windowsize
        self.framecache = []  # list of frames for each game, cleared at the end of the game
        self.gameIter = 0
        self.bestScore = 0
        self.maxCache = options.maxCache
        self.isTrain = options.isTrain
        self.env = env
        self.actions = ['Right', 'Left', 'A', ['Right', 'A'], ['Right', 'B'], ['Right', 'A', 'B'], ['Left', 'A'],
                        ['Left', 'B'], ['Left', 'A', 'B']]
        self.algo = QLearningAlgorithm(
            options=options,
            actions=self.actions,
            discount=0.9,
            featureExtractor=self.featureExtractor
        )
        self.stepCounter = 0
        self.stepCounterMax = 4
        self.totalReward = 0

    def featureExtractor(self, window, action):
        raise Exception('Abstract method! should be overridden')

    def initAction(self):
        return self.action

    def cacheState(self):
        frames = self.framecache[-self.windowsize:]
        last_frame = frames[-1]
        last_frame = last_frame.set_reward(self.totalReward)
        state = GameState(frames[:-1] + [last_frame])
        self.totalReward = 0
        self.algo.statecache[-1].append(state)
        return state

    def act(self, obs, reward, is_finished, info):
        self.frame = GameFrame(np.copy(obs), reward, is_finished, info.copy())
        self.totalReward += reward

        # caching new frame
        self.framecache.append(self.frame)

        if len(self.framecache) > self.windowsize:
            self.framecache.pop(0)  # remove frame outside window to save memory

        # only update state and action once a while
        self.stepCounter += 1
        if self.stepCounter < self.stepCounterMax:
            if is_finished:
                self.cacheState()
            return self.action

        # if not enough frame for a window, keep playing 
        if len(self.framecache) < self.windowsize:
            return self.action

        self.stepCounter = 0

        # caching new state
        state = self.cacheState()

        self.algo.incorporateFeedback()

        # get new action
        if is_finished:
            self.action = Action.empty()
        else:
            # get and cache new action 
            self.action, action_idx = self.algo.getAction(state)
            self.algo.actioncache[-1].append(action_idx)

        self.log(self.action, reward)
        return self.action

    def exit(self):
        if self.frame is None:
            return False
        if self.frame.get_is_finished():
            frame_info= self.frame.get_info()
            if 'distance' in frame_info:
                distance = frame_info['distance']
                if self.bestScore < distance:
                    self.bestScore = distance
                    print "Best Score: {}".format(self.bestScore)
                print "Score: {}".format(distance)
            self.gameIter += 1
            self.env.reset()
            self.framecache = []
            self.algo.actioncache.append([])
            self.algo.statecache.append([])
            self.totalReward = 0

            if len(self.algo.statecache) > self.maxCache:
                self.algo.actioncache.pop(0)
                self.algo.statecache.pop(0)

        info = self.frame.get_info()
        stuck = False
        if 'ignore' in info:
            stuck = info['ignore']

        reachMaxIter = self.gameIter >= self.maxGameIter

        exit = reachMaxIter
        exit = exit and (self.frame.get_is_finished() or stuck)
        return exit

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)
