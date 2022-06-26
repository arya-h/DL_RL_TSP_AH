from agentTSP import AgentTSP
from envTSP import TSPDistCost

if __name__ == "__main__":

    test_env = TSPDistCost()
    agent = AgentTSP()
    best_reward = -4000
    print(agent.env.distance_matrix)
    # perform N steps to fill reward & transitions tables
    agent.play_n_random_steps(1000000)
    print("I have finished playing my random steps")
    # run value iteration over all states
    agent.value_iteration()
    reward = 0.0

    while True:
        newReward, path = agent.play_episode(test_env)
        if(best_reward < 0 and newReward > best_reward):
            best_reward = newReward
            if (best_reward > 0) :
                print(f"{newReward} --> {path}")
                agent.env.render_custom(path)
                display_distances(agent.env.distance_matrix, path)
                best_reward = newReward
                continue

        elif newReward < best_reward and newReward>0:
            best_reward = newReward
            print(f"{newReward} --> {path}")
            display_distances(agent.env.distance_matrix, path)
            agent.env.render_custom(path)