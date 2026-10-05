from collections import OrderedDict
from nn.q_model import *
from QLearnAgent import *
import numpy as np
from FeatureExtractor import *


class ManualFeatureAgent(QLearnAgent):
    def __init__(self, options, env):
        super(ManualFeatureAgent, self).__init__(options, env)
        self.featureExtractors = []
        self.featureExtractors.append(InfoFeatureExtractor())
        self.featureExtractors.append(VelocityFeatureExtractor())
        self.featureExtractors.append(MarioFeatureExtractor())
        self.featureExtractors.append(FrontFeatureExtractor())
        self.featureExtractors.append(PitFeatureExtractor())
        self.featureExtractors.append(BehindFeatureExtractor())
        self.featureExtractors.append(EnemyFeatureExtractor())
        self.featureExtractors.append(PrevActionsFeatureExtractor(options, self.actions))
        self.featureExtractors.append(Height5FeatureExtractor())

        featureSize = sum([fe.featureSize() for fe in self.featureExtractors])
        print('featureSize', featureSize)
        self.model = QModel(
            options=options,
            info_size=featureSize,
            num_actions=len(self.actions),
            tile_row=1,
            tile_col=1,
            window_size=self.windowsize,
            conv=False
        )
        self.algo.set_model(self.model)

    def featureExtractor(self, window):
        feature = OrderedDict()
        for fe in self.featureExtractors:
            fe.extract(feature, window)
        return feature
