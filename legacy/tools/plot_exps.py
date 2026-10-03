import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import path
from plot_scores import *

def plot(options, labels, name, xlim=None, legendloc=None):
    fig, ax = plt.subplots()

    for path in labels:
        ax.plot(options.indices[path][options.smooth_factor:], options.scores[path],
                label=labels[path])
        ax.set(xlabel='Game Number', ylabel='Distance',
               title='Average distance Mario reaches during training')
    ax.grid()

    if xlim is not None:
        ax.set_xlim(xlim)

    if legendloc is None:
        legendloc = 'upper left'
    legend = ax.legend(loc=legendloc, shadow=True)
    fig.savefig('figs/' + name + ".png")
    plt.show()

def plot_exprate_comp(options):
    labels = {}
    labels['model/20171213_164419'] = 'manual-adaptive'
    labels['model/20171213_165416'] = 'manual-fixed (0.3)'
    labels['model/20171212_182429'] = 'cnn-adaptive'
    labels['model/20171214_161615'] = 'cnn-fixed (0.3)'

    plot(options, labels, 'exprate_comp', xlim=None)
    return options

def plot_reward_comp(options):
    labels = {}
    labels['model/20171214_160447'] = 'distance_only'
    labels['model/20171212_182429'] = 'distance+death+stuck'
    labels['model/20171213_021909'] = 'distance+death+stuck+ckpt'

    plot(options, labels, 'reward_comp', xlim=None)
    return options

def plot_batch_comp(options):
    labels = {}
    labels['model/20171212_182429'] = 'batch=20 #Batch=11 updateInterval=10'
    labels['model/20171212_181553'] = 'batch=64 #Batch=16 updateInterval=64'
    labels['model/20171212_210533'] = 'batch=128 #Batch=16 updateInterval=256'

    plot(options, labels, 'batch_comp', xlim=None)

    return options

def plot_lr_comp(options):
    labels = {}
    labels['model/20171212_182429'] = 'lr=1e-4'
    labels['model/20171212_182700'] = 'lr=1e-3'
    labels['model/20171212_182911'] = 'lr=1e-2'

    plot(options, labels, 'lr_comp')

    return options

def plot_arch_comp(options):
    labels = {}
    # labels['saved_model/manual'] = 'manual nn'
    labels['model/20171213_164419'] = 'manual nn'
    labels['saved_model/CNN_action'] = '[7x7-s3] cnn'
    labels['saved_model/best_model'] = '[5x5-s2] + [3x3-s1] cnn'
    labels['model/20171212_182429'] = '2 x [3x3-s1] cnn'

    plot(options, labels, 'arch_comp', xlim=[0, 7000], legendloc='upper right')

    return options

def plot_all(options):
    labels = {}

    for path in options.paths:
        labels[path] = path

    plot(options, labels, 'all_exp')

    return options

def load_scores(options):
    indices = {}
    scores = {}

    for path in options.paths:
        indices[path], scores[path] = load_score(path + '/score_log')

        scores[path] = smoothByAverage(scores[path], options.smooth_factor)

    options.indices = indices
    options.scores = scores
    return options

def set_paths(options):
    paths = []
    paths.append('saved_model/CNN_action')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/
    
    # Cannot load parameter for some reason
    paths.append('model/20171210_154650')
    #./xvfb-run-safe -s "-screen 1 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/
    
    paths.append('model/20171212_181553')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--batchSize=64 \
      #--batchPerFeedback=16 \
      #--updateInterval=64 \
      #--load --model_dir $MODEL/
    
    paths.append('model/20171212_210533')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--batchSize=128 \
      #--batchPerFeedback=16 \
      #--updateInterval=256 \
      #--save_period=500 \
      #--load --model_dir $MODEL/
    
    #tucson
    paths.append('model/20171212_182429')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/
    
    paths.append('model/20171212_182700')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--lr=1e-3 \
      #--load --model_dir $MODEL/
    
    paths.append('model/20171212_182911')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--maxGameIter=3000 \
      #--lr=1e-2 \
      #--load --model_dir $MODEL/
    
    # tucson
    paths.append('model/20171213_021909')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--partial_reward \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/

    # tucson
    paths.append('model/20171214_160447')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--dist_reward_only \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/

    # tucson
    paths.append('model/20171213_164419')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=manual --train \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/
    
    # tucson
    paths.append('model/20171213_165416')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=manual --train \
      #--fix_exprate \
      #--explorationProb=0.3 \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/

    # tucson
    paths.append('model/20171214_161615')
    #./xvfb-run-safe -s "-screen 0 1400x900x24" python run.py --player=cnn --train \
      #--conv_model=1 \
      #--explorationProb=0.3 \
      #--maxGameIter=3000 \
      #--load --model_dir $MODEL/

    paths.append('saved_model/manual')
    paths.append('saved_model/best_model')
    options.paths = paths
    return options

def main():
    usage = "Usage: run [options]"
    parser = argparse.ArgumentParser()

    # Game options
    parser.add_argument('--smooth_factor', dest='smooth_factor', action='store', nargs='?',
            type=int, default=20, help='number of points to average over')

    (options, args) = parser.parse_known_args()

    options = set_paths(options)
    options = load_scores(options)

    options = plot_all(options)
    options = plot_arch_comp(options)
    options = plot_lr_comp(options)
    options = plot_batch_comp(options)
    options = plot_reward_comp(options)
    options = plot_exprate_comp(options)

if __name__ == "__main__":
    main()
