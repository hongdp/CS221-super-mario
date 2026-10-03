from enum import *
import numpy as np
from collections import OrderedDict



def out_of_frame(x, y):
    return x < 0 or y < 0 or x >= Window.Width or y >= Window.Height


# get tile at (mario.x + step_x, mario.y + step_y)
# where positive step_x and step_y indicates right and up
# obs is 13 x 16 (y, x) with (0,0) on top left corner
def get_tile_from_mario(obs, step_x, step_y):
    focusy, focusx = get_coord_from_mario(obs, step_x, step_y)
    return obs[focusy, focusx]


def get_mario_coord(obs):
    ys, xs = np.where(obs == Tile.MARIO)
    if len(xs) == 0:
        return None
    mariox = xs[0]
    marioy = ys[0]
    return marioy, mariox


def get_coord_from_mario(obs, step_x, step_y):
    coord = get_mario_coord(obs)
    if coord is None:
        return None
    marioy, mariox = coord
    focusx = mariox + step_x
    focusy = marioy - step_y
    if out_of_frame(focusx, focusy):
        return None
    return focusy, focusx


class GameState(object):
    def __init__(self, frames, prev_actions):
        self.frames = frames
        self.prev_actions = prev_actions

    def num_frames(self):
        return len(self.frames)

    def get_frames(self):
        return self.frames

    def get_last_frame(self):
        return self.frames[-1]

    def get_prev_actions(self):
        return self.prev_actions

    def get_last_n_obs(self, n=1):

        obs = [frame.get_obs() for frame in self.frames[-n:]]
        # obs = [[np.zeros([Window.Width, Window.Height])]] * (n - len(self.frames))
        # for i in range(min([n, len(self.frames)])):
            # obs.append(self.frames[-1-i].get_obs())
        return obs


class GameFrame(object):
    def __init__(self, obs, reward, is_finished, info):
        self.obs = obs
        self.reward = reward
        self.is_finished = is_finished
        self.info = info

    def get_reward(self):
        return self.reward

    def set_reward(self, newReward):
        self.reward = newReward
        return self

    def get_obs(self):
        return self.obs

    def get_is_finished(self):
        return self.is_finished

    def get_info(self):
        return self.info

class GameAction(object):
    def __init__(self, actions):
        self.actions = actions

        count = OrderedDict()
        for act in self.actions:
            if tuple(act) not in count:
                count[tuple(act)] = 0
            count[tuple(act)] += 1
        self.name = '_'.join(['{}-{}'.format('-'.join(acts), count[tuple(acts)]) for acts in
            count.keys()]) 

    def get_actions(self):
        return self.actions[:]

    def __str__(self):
        return self.name

# get mario's velocity from state1 to state2
def get_velocity(state1, state2):
    # 13 x 16 numpy array
    obs1 = state1.get_obs()
    obs2 = state2.get_obs()
    min_diff = float('inf')
    for v in range(3):
        shifted_obs1 = np.roll(obs1, v, axis=1)
        shifted_obs1[:, :v] = obs2[:, :v]
        diff = np.sum(np.abs(shifted_obs1 - obs2))
        if diff < min_diff:
            min_diff = diff
            min_v = v
    return v


def get_death_penalty_value(time, distance):
    return (time - Time.TOTAL_GAME_TIME) - distance / 3

def get_stuck(frames):
    dists = [frame.get_info()['distance'] for frame in frames]
    if len(set(dists)) == 1: # dists are the same
        return True
    return False
