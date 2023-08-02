import gym
from agents.machine_agent import MachineAgent


def get_agent_score_cartpole(agent, env, close_on_finish=True, n_trials=1, max_steps=5000):
    if env is None:
        env = gym.make('CartPole-v1')

    tot_score = 0
    for _ in range(n_trials):
        obs = env.reset()[0]
        done = False
        score = 0
        steps = 0
        while not done:
            action = agent.get_action(obs)
            obs, reward, done, info, _ = env.step(action)
            score += reward
            steps += 1
            if steps > max_steps:
                break

        tot_score += score

    if close_on_finish:
        env.close()

    return tot_score / n_trials


def get_fitness_cartpole(population, max_steps=5000):
    scores = []
    env = gym.make('CartPole-v1')

    for network in population:
        agent = MachineAgent(network, action_size=2,
                             observation_size=4, problem_type='cartpole')
        score = get_agent_score_cartpole(agent, env, False, 1, max_steps)

        scores.append(score)
    return scores
