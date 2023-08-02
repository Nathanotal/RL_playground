import gym
from evaluation.cartpole import get_agent_score_cartpole
from evaluation.mountaincar import get_agent_score_mountaincar

def visualize_agent(agent, max_steps=1000, type = 'cartpole'):
    if type == 'cartpole':
        env = gym.make('CartPole-v1', render_mode='human')
    elif type == 'mountaincar_cont':
        env = gym.make('MountainCarContinuous-v0', render_mode='human')
        
    obs = env.reset()[0]
    done = False
    score = 0
    step = 0
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, info, _ = env.step(action)
        score += reward
        step += 1
        if step > max_steps:
            break

    env.close()
    print("Score:", score)

def compare_agents(agents, type = 'cartpole'):
    if type == 'cartpole':
        get_agent_score = get_agent_score_cartpole
    elif type == 'mountaincar_cont':
        get_agent_score = get_agent_score_mountaincar
        
    agent_to_score = {}
    for agent in agents:
        score = get_agent_score(agent, n_trials=10)
    
    # Sort by score
    sorted_agents = sorted(agent_to_score.items(), key=lambda x: x[1], reverse=True)
    # Pretty print
    print("🏆🏆🏆 Leaderboard 🏆🏆🏆")
    print("-------------------------------")
    print("Rank | Name          | Score")
    print("-------------------------------")
    rank = 1
    for agent, score in sorted_agents:
        medal = "    "
        if rank == 1:
            medal = "🥇 "
        elif rank == 2:
            medal = "🥈 "
        elif rank == 3:
            medal = "🥉 "

        name = agent.name
        # Adjust the following number to match the longest agent name in your list
        padding = 15 - len(name)
        print(f"{medal} {rank: <4}| {name: <{padding + len(name)}}| {score}")
        rank += 1

    print("-------------------------------")