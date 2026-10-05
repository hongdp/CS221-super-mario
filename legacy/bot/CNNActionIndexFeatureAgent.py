from collections import OrderedDict
import numpy as np
from nn.q_model import *
from QLearnAgent import *
from FeatureExtractor import *


class CNNActionIndexFeatureAgent(QLearnAgent):
    def __init__(self, options, env):
        super(CNNActionIndexFeatureAgent, self).__init__(options, env)
        self.featureExtractors = []
        self.featureExtractors.append(InfoFeatureExtractor())
        self.featureExtractors.append(VelocityFeatureExtractor())
        self.featureExtractors.append(MarioFeatureExtractor())
        self.featureExtractors.append(EnemyFeatureExtractor())
        self.featureExtractors.append(PrevActionIndexFeatureExtractor(options, self.actions))

        self.tileFeatureExtractor = TileFeatureExtractor(options)

        featureSize = sum([fe.featureSize() for fe in self.featureExtractors])
        tile_row, tile_col, window_size = self.tileFeatureExtractor.featureSize()
        self.model = QModel(
            options=options,
            info_size=featureSize,
            tile_row=tile_row,
            tile_col=tile_col,
            window_size=window_size,
            num_actions=len(self.actions)
        )
        self.algo.set_model(self.model)

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
    def featureExtractor(self, window):
        feature = OrderedDict()
        for fe in self.featureExtractors:
            fe.extract(feature, window)
        tiles = self.tileFeatureExtractor.extract(OrderedDict(), window)
        return tiles, feature
