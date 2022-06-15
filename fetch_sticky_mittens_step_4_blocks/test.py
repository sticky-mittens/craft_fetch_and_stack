import gym
import gym_fetch_stack
from IPython import display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from uber_policies import action_policies

from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make("FetchStack2Stage1-v1")
# env = gym.wrappers.Monitor(env, './media/',video_callable=lambda episode_id: True,force = True)
env = gym.wrappers.Monitor(env, './media/',force = True)

# video_recorder = None
# video_recorder = VideoRecorder(env, "./media/saved.mp4", enabled = True)

obs = env.reset()
done = False
img = plt.imshow(env.render(mode='rgb_array'))

def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()

while not done:
    env.render()

    # video_recorder.capture_frame()
    # action = policy(obs["observation"], obs["desired_goal"])


    # action = action_policies.implement_option("object_1", obs)
    action = action_policies.compute_option_1(obs)
    obs, reward, done, info = env.step(action)
    print("observation - ", obs["annotated_obs"]["object_0"]["rel_pos"])
    print("Goal - ", obs["desired_goal"])


# video_recorder.close()
# video_recorder.enabled = False
env.close()
