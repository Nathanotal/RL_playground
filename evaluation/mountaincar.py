import gym
from agents.machine_agent import MachineAgent
import math


def get_agent_score_mountaincar(agent, env, close_on_finish=True, n_trials=1, max_steps=5000):
    if env is None:
        env = gym.make('MountainCarContinuous-v0')

    tot_score = 0
    for _ in range(n_trials):
        obs = env.reset()[0]
        done, terminated = False, False
        score = 0
        steps = 0
        max_height = 0
        while not done and not terminated:
            action = agent.get_action(obs)
            obs, reward, terminated, done, info = env.step(action)
            score += reward
            steps += 1
            if steps > max_steps:
                break

        tot_score += score

    if close_on_finish:
        env.close()

    return tot_score / n_trials


def get_fitness_mountaincar(population, max_steps=500):
    scores = []
    env = gym.make('MountainCarContinuous-v0')

    for network in population:
        agent = MachineAgent(network, action_size=1,
                             observation_size=2, problem_type='mountaincar_cont')
        score = get_agent_score_mountaincar(agent, env, False, 1, max_steps)

        scores.append(score)
    return scores
