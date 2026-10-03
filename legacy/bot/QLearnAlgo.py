from numpy import random
import operator
from util import *

class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor):
        # hyper parameter
        self.updateInterval = options.updateInterval 
        self.updateTargetInterval = options.updateTargetInterval
        self.batchSize = options.batchSize 
        self.batchPerFeedback = options.batchPerFeedback
        self.maxCache = options.maxCache
        self.explorationProb = options.explorationProb

        # inits
        self.actions = actions
        self.options = options
        self.featureExtractor = featureExtractor
        self.discount = discount
        self.updateCounter = 0
        self.updateTargetCounter = 0
        self.statecache = [[]]  # list of states for each game. A state is a window of frames
        self.actioncache = [[]]  # list of actions for each game
        self.model = None
        self.softmaxExplore = options.softmaxExploration

    # reset after each game iteration
    def reset(self):
        self.actioncache.append([])
        self.statecache.append([])
        # boost explorationProb slightly at beginning of the next game
        if self.options.isTrain and not self.options.fix_exprate:
            self.explorationProb = min(self.explorationProb * 1.2, 0.5)
        if len(self.statecache) > self.maxCache:
            self.actioncache.pop(0)
            self.statecache.pop(0)

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, vs, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            info = info.values()
            scores = self.model.inference_Q(vs, [info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [0]
            info = info.values()
            scores = self.model.inference_Q(vs, [info])[0]
        return scores

    def getProb(self, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            info = info.values()
            scores = self.model.inference_Prob([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [1.0 / len(self.actions)] * len(self.actions)
            info = info.values()
            scores = self.model.inference_Prob([info])[0]
        return scores

    def formatQ(self, Q):
        info = 'Q: '
        for i, q in enumerate(Q):
            info += '{} {:.2f}, '.format(str(self.actions[i]), q) 
        return info

    def formatProb(self, prob):
        info = 'Prob: '
        for i, p in enumerate(prob):
            info += '{} {:.3e}, '.format(str(self.actions[i]), p) 
        return info

    def debugState(self, state):
        # debug print
        if self.model.conv:
            tile, info = self.featureExtractor(state)
        else:
            info = self.featureExtractor(state)
        show = []
        # show += ['pit-ahead']
        infostr = ''
        for k in show:
            infostr += k + '=' + str(info[k]) + ', '
        print(infostr)

        # prevAction = Action.names([info['prevActions-5-Bit{}'.format(i)] for i in range(6)])
        # print('prevAction', prevAction)

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.debugState(state)

        msg = ""
        actionIdx = 0

        if self.options.isTrain and random.random() < self.explorationProb:
            actionIdx = random.choice(range(len(self.actions)))
            msg += "randomly select action: {}".format(self.actions[actionIdx])
        else:
            if self.softmaxExplore:
                prob = self.getProb(state)
                actionIdx = random.choice(range(len(self.actions)), p=prob)
                msg += self.formatProb(prob)
                msg += "\nsoftmax action: {}".format(self.actions[actionIdx])
            else:
                q = self.getQ(self.model.prediction_vs, state)
                actionIdx, _ = max(enumerate(q), key=operator.itemgetter(1))
                msg += self.formatQ(q)
                msg += "\nMax action: {}".format(self.actions[actionIdx])

        msg += " exploreProb={}".format(self.explorationProb)
        print(msg)
        # decay explorationProb over game
        if self.options.isTrain and not self.options.fix_exprate:
            self.explorationProb = max(0.1, self.explorationProb * 0.98)

        return actionIdx

    # Call this function to get the step size to update the weights.

    def sample(self, sampleSize):
        # randomly choose a game and get its states
        samples = []
        gameLenSum = sum([len(a) for a in self.actioncache])
        gameProb = [len(a) * 1.0 / gameLenSum for a in self.actioncache]
        for i in range(sampleSize):
            gameIdx = random.choice(range(0, len(self.statecache)), p=gameProb)
            # Should have cache of s0, a0, ....., sn, an, sn+1, where reward of an is stored in sn+1
            gameStates = self.statecache[gameIdx]
            gameActions = self.actioncache[gameIdx]

            if len(gameActions) == 0:
                return self.sample()  # resample a different game

            # randomly choose a state except last one in the game
            stateIdx = random.randint(0, len(gameActions)) if len(gameActions) > 1 else 0

            state_n = gameStates[stateIdx]
            if self.model.conv:
                tile, info = self.featureExtractor(state_n)
                info = info.values()
            else:
                tile = None
                info = self.featureExtractor(state_n)
                info = info.values()
            action = gameActions[stateIdx]

            state_np1 = gameStates[stateIdx + 1]
            reward = state_np1.get_last_frame().get_reward()
            Vopt = max(self.getQ(self.model.target_vs, state_np1))
            gamma = self.discount
            target = (reward + gamma * Vopt)
            if state_np1.get_last_frame().get_is_finished():
                target = reward
                print('sampled last state, action', str(self.actions[action]), 'reward', reward)

            samples.append((tile, info, action, target))

        return samples

    # once a while train the model
    def incorporateFeedback(self):
        if not self.options.isTrain: return

        self.updateCounter += 1
        if self.updateCounter < self.updateInterval:
            return
        self.updateCounter = 0

        print('incorporateFeedback ...')
        for i in range(self.batchPerFeedback):
            tiles = []  # a list of None if self.mode.conv is False
            infos = []
            actions = []
            target_Qs = []
            samples = self.sample(self.batchSize)
            for tile, info, action, target in samples:
                tiles.append(tile)
                infos.append(info)
                actions.append(action)
                target_Qs.append(target)

            self.model.update_weights(tiles=tiles, infos=infos, actions=actions, target_Qs=target_Qs)

        self.updateTargetCounter += 1
        if self.updateTargetCounter < self.updateTargetInterval:
            return
        self.updateTargetCounter = 0
        print('Updating Target Network ...')
        self.model.update_target_network()
