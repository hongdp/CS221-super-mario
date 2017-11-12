import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
from QLearnAlgo import *

class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = [0,0,0,0,0,0]
        self.state = None
        self.maxGameIter = options.maxGameIter
        self.windowsize = 3
        self.gameIter = 0
        self.isTrain = options.isTrain
        self.env = env
        self.algo = QLearningAlgorithm(
                options = options,
                actions = self.get_possible_actions,
                discount = 1,
                featureExtractor = self.featureExtractor,
                windowsize = self.windowsize,
                explorationProb = 0.4
                )

    def get_possible_actions(self, state):
        return ['Left', 'Right', 'A', 'B'] + [Action.NO_ACTION]

    def featureExtractor(self, window, action):
        raise Exception('Abstract method! should be overritten')

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)

        if len(self.algo.statecache) >= 1:
            prevState = self.algo.statecache[-1]
            prevAction = self.algo.actioncache[-1]
            self.algo.incorporateFeedback(prevState, prevAction, self.state)

        self.action = self.algo.getAction(self.state)

        # caching states
        self.algo.statecache.append(self.state)
        self.algo.actioncache.append(self.action)

        self.logAction()
        return self.action

    def exit(self):
        if self.state is None:
            return False
        if is_finished(self.state):
            self.gameIter += 1
            self.env.reset()
            self.algo.actioncache = []
            self.algo.statecache = []
            self.algo.batchcounter = 0

        info = get_info(self.state)
        stuck = False
        if 'ignore' in info:
            stuck = info['ignore']

        reachMaxIter = self.gameIter >= self.maxGameIter

        exit = (reachMaxIter and self.isTrain) or not self.isTrain
        exit = exit and (is_finished or stuck)
        return exit

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)