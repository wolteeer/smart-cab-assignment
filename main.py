import gym
from IPython.display import clear_output
from time import sleep
import numpy as np

env = gym.make("Taxi-v3").env  # set game environment

frames = []  # for animation

trained_frames = []

q_table = np.zeros([env.observation_space.n, env.action_space.n])  # q-table with zero's


# Get action space and observation space for basic understanding of the env
def get_env_statistics():
    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))


# Manipulate the environment sketch a desired environment.
# It wil also render the env so you can see what you've done
def manipulate_env(taxi_row, taxi_column, passenger_index, destination_index):
    state = env.encode(taxi_row, taxi_column, passenger_index, destination_index)
    env.s = state
    env.render()
    print(env.P[state])  # this will sketch the Reward Table


# Brute force our illustration
def brute_force_smart_cab():
    env.s = 328  # set environment to illustration's state
    epochs = 0
    penalties, reward = 0, 0

    done = False

    while not done:
        action = env.action_space.sample()  # choose random action from action space
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

        epochs += 1

    print("Timesteps taken: {}".format(epochs))
    print("Penalties incurred: {}".format(penalties))


# Helpful function to print frames IFF frames is filled.
def print_frames(frames):
    for i, frame in enumerate(frames):
        clear_output(wait=True)
        print(frame['frame'])
        print("Timestep: %s" % ((i + 1),))
        print("State: %s" % (frame['state'],))
        print("Action: %s" % (['action'],))
        print("Reward: %s" % (frame['reward'],))
        sleep(.1)


def dqn_train():
    # Hyperparameters
    alpha = 0.0  # play around with these
    gamma = 0.0  # play around with these
    epsilon = 0.0  # play around with these

    # For plotting metrics
    all_epochs = []
    all_penalties = []

    # we train 100,000 episodes
    for i in range(1, 100001):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        done = False

        while not done:
            ###
            ### Put your algorithm here
            ###

            if reward == -10:
                penalties += 1

            # state = next_state ### uncomment when algorithm is implemented
            epochs += 1

    print("Training finished.\n")


def evaluate_agent():
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0

        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1

            trained_frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward
            }
            )

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print("Results after %s episodes:" % (episodes,))
    print("Average timesteps per episode: %s" % (total_epochs / episodes,))
    print("Average penalties per episode: %s" % (total_penalties / episodes,))


def evaluate_agent_on_illustration():
    env.s = 328  # set environment to illustration's state
    epochs, epochs, penalties = 0, 0, 0

    done = False

    state = env.s

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1

        trained_frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
        }
        )

        epochs += 1

    print("Average timesteps per episode: %s" % (epochs))
    print("Average penalties per episode: %s" % (penalties))


# manipulate_env(3, 1, 2, 0) try it out

# brute_force_smart_cab() -> uncomment next line for visual result.
# print_frames(frames)

dqn_train()

# q_table[328]  # check which move is suggested by the q-table, in the illustration state.

# evaluate_agent_on_illustration()
# print_frames(trained_frames)
