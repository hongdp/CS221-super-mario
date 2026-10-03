import gym
import ppaquette_gym_super_mario
from ppaquette_gym_super_mario.wrappers.control import *
import argparse
from bot import *

def print_options(options):
    optionDict = vars(options)
    for k in optionDict:
        print(k + ' = ' + str(optionDict[k]))

def load_options(options):
    print('Loading options ...')
    new_options = options
    option_path = options.model_dir + '/options.pickle'
    if not os.path.isfile(option_path):
        print('No parameters stored in {}'.format(option_path))
        exit(-1)
    options = pickle.load(open(option_path, 'rb'))
    options.load = new_options.load
    options.isTrain = new_options.isTrain
    options.model_dir = new_options.model_dir
    options.ckpt = new_options.ckpt
    options.maxGameIter = new_options.maxGameIter
    options.batchSize = new_options.batchSize
    options.batchPerFeedback = new_options.batchPerFeedback
    options.updateInterval = new_options.updateInterval
    options.updateTargetInterval = new_options.updateTargetInterval
    options.save_period = new_options.save_period

    if not hasattr(options, 'conv_model'):
        options.conv_model = 0

    if not hasattr(options, 'partial_reward'):
        options.partial_reward = False

    if not hasattr(options, 'fix_exprate'):
        options.fix_exprate = False

    if not hasattr(options, 'dist_reward_only'):
        options.dist_reward_only = False

    if options.ckpt < 0:
        for d in [x for x in os.listdir(options.model_dir)]:
            if 'ckpt' in d:
                ckpt = int(d.split('ckpt')[1])
                if options.ckpt < ckpt:
                    options.ckpt = ckpt
        if options.ckpt < 0:
            print('No ckpt in {}!'.format(options.model_dir))
            exit(-1)
        else:
            print('Loading latest ckpt{}...'.format(options.ckpt))

    return options

def create_agent(options, env):
    if options.player == 'human':
        agent = HumanAgent(options)
        wrapper = SetPlayingMode('human')
        env = wrapper(env)
        options.isTrain = False
    elif options.player == 'baseline':
        agent = BaselineAgent(options)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
        options.isTrain = False
    elif options.player == 'cnn':
        agent = CNNFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'cnnidx':
        agent = CNNActionIndexFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'feature':
        agent = FeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    elif options.player == 'manual':
        agent = ManualFeatureAgent(options, env)
        wrapper = SetPlayingMode('algo')
        env = wrapper(env)
    return agent,env

def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()

    # Game options 
    parser.add_argument('--model_dir', dest='model_dir', action='store', default='./model',
            help='Directory to store weights')
    parser.add_argument('--player', dest='player', action='store', default='human',
                        help='Specify the player, valid option: human, baseline, feature')
    parser.add_argument('--no-gui', dest='render', action='store_false', default=True,
                        help='Do not render visualization of the game')
    parser.add_argument('--train', dest='isTrain', action='store_true', default=False,
                        help='Training mode')
    parser.add_argument('--load', dest='load', action='store_true', default=False,
                        help='load weights')
    parser.add_argument('--ckpt', dest='ckpt', nargs='?', default=-1, type=int, help='ckpt number of a training')

    # Game hyper parameter
    parser.add_argument('--maxGameIter', dest='maxGameIter', nargs='?', default=1, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--stepCounterMax', dest='stepCounterMax', nargs='?', default=6, type=int,
                        help='Number of frames to advance state')
    parser.add_argument('--updateInterval', dest='updateInterval', nargs='?', default=10, type=int,
                        help='Number of frames to retrain the model')
    parser.add_argument('--updateTargetInterval', dest='updateTargetInterval', nargs='?', default=20, type=int,
                        help='Number of updates to update the target network')
    parser.add_argument('--maxCache', dest='maxCache', nargs='?', default=1000, type=int,
                        help='Max number of training iteration')
    parser.add_argument('--softmaxExploration', dest='softmaxExploration', action='store_true', default=False,
                        help='Exploration mode')
    parser.add_argument('--window', dest='windowsize', nargs='?', default=3, type=int,
                        help='Number of frames to include in a state')
    parser.add_argument('--tileWindowSize', dest='tileWindowSize', nargs='?', default=3, type=int,
                        help='Number of frames the TileFeatureExtractor extracts')
    parser.add_argument('--prevActionsSize', dest='prevActionsSize', nargs='?', default=1, type=int,
                        help='Number of previous action to include in states')
    parser.add_argument('--batchSize', dest='batchSize', nargs='?', default=20, type=int,
                        help='Number of samples to train the model')
    parser.add_argument('--batchPerFeedback', dest='batchPerFeedback', nargs='?', default=11, type=int,
                        help='Number of batched updates before continue playing')
    parser.add_argument('--explorationProb', dest='explorationProb', nargs='?', default=0.5,
            type=float, help='Exploration Probability. Decay over time')
    parser.add_argument('--fix_exprate', dest='fix_exprate', action='store_true', default=False,
            help='turn off adaptively adjusting exploration rate')
    parser.add_argument('--death_penalty', dest='death_penalty', nargs='?', default=-100,
            type=int, help='Death penalty to give if gets killed')
    parser.add_argument('--partial_reward', dest='partial_reward', action='store_true', default=False,
                        help='Enable partial reward')
    parser.add_argument('--dist_reward_only', dest='dist_reward_only', action='store_true', default=False,
                        help='Only use distance as reward')

    # Model hyper parameters
    parser.add_argument('--conv_model', dest='conv_model', nargs='?', default=0, type=int,
            help='Specify which conv architecture to use')
    parser.add_argument('--optimizer', dest='optimizer', action='store', default='adam', help='SGD optimizer')
    parser.add_argument('--lr', dest='lr', nargs='?', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--decay_step', dest='decay_step', nargs='?', default=1000, type=int,
            help='decay step')
    parser.add_argument('--decay_rate', dest='decay_rate', nargs='?', default=1, type=int,
            help='decay rate')
    parser.add_argument('--regularization', dest='regularization', nargs='?', default=0.001, type=float,
            help='regularization strength')
    parser.add_argument('--gradient_clip', dest='gradient_clip', nargs='?', default=10, type=int,
            help='gradient_clip')
    parser.add_argument('--save_period', dest='save_period', nargs='?', default=2000, type=int,
            help='save_period')

    (options, args) = parser.parse_known_args()

    env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

    if options.load: # testing. loading options
        options = load_options(options)
    print_options(options)

    if options.isTrain and not options.load:
        options.model_dir = "model/{:%Y%m%d_%H%M%S}".format(datetime.now())

    if not os.path.exists(options.model_dir):
        os.makedirs(options.model_dir)

    agent,env = create_agent(options, env)

    env.reset()

    print('Started game ...')
    action = agent.initAction()
    while not agent.exit():
        try:
            obs, reward, is_finished, info = env.step(action)
            if options.render:
                env.render()

            action = agent.act(obs, reward, is_finished, info)
        except Exception as e:
            agent.handle(e)

if __name__ == "__main__":
    main()
