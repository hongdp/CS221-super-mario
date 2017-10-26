import gym
import ppaquette_gym_super_mario
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
env.reset()
total_score = 0
first = True
while total_score < 32000:
    action = [0,0,0,1,0,0]
    obs, reward, is_finished, info = env.step(action)
    if first:
        first = False
        print obs
    env.render()
    total_score = info["distance"]
    print total_score