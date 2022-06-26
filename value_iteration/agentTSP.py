import collections
import pickle
from envTSP import TSPDistCost

GAMMA = 0.9

class AgentTSP:
    def __init__(self):
        #set the environment
        #self.env = or_gym.make(ENV_NAME)
        self.env = TSPDistCost()
        #reset the state
        self.state = self.env.reset()
        # rewards table
        self.rewards = collections.defaultdict(int)
        # transitions table
        self.transits = collections.defaultdict()
        # value table
        self.values = collections.defaultdict(float)

    # func used to gather random experience, update reward & transitions table
    # there is no need to wait for the end of the entire episode to learn

    def play_n_random_steps(self, count):
        for _ in range(count):
            if (_%100000 == 0):
                print(f"{_} / {count}")
            action = self.env.action_space.sample()
            state = self.state.tolist()
            state_tuple = tuple(state)
            new_state, reward, is_done, _ = self.env.step(action)
            new_state_tuple = tuple(new_state.tolist())

            as_bytes_newstate = new_state.tobytes()
            self.rewards[(state_tuple, action, new_state_tuple)] = reward

            self.transits[(state_tuple, action)] = new_state_tuple
            self.state = self.env.reset() if is_done else new_state

    #calculates the value of the action from the state using
        # --transition table
        # --reward table
        # --values table

    #two purposes
    # 1. select the best action to perform from the state
    # 2. calculate the new value of the state on value iteration.
    def calc_action_value(self, state, action):
        #extract transition counters for the given (state,action) tuple
        #from the transitions table
        #dict has
        # --KEY : target state

        action_value = 0.0
        #target state is single, deterministic environment
        tgt_state = self.transits[state, action]

        reward = self.rewards[(state, action, tgt_state)]
        #calculate the updated action value with bellman equation
        #nb, Q(s) =  (probability of landing in that state)*[immediate reward + discounted value for the target state]
        action_value += (reward + GAMMA * self.values[tgt_state])
        return action_value

    # decides the best action to take from a given state
    # it iterates over all possible actions in the env and calculates
    # the value for every action.
    # it returns the action with the largest value, which will be chosen
    def select_action(self, state):
        #best_value set to 50000 to guarantee it excludes negative values in the if
        best_action, best_value = None, 50000
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value > action_value and action_value>0:
                best_value = action_value
                best_action = action
        return best_action

    # uses the select_action function to choose the best action to take
    # it plays one full episode using the provided environment
    # used to play TEST EPISODES, in order not to influence the current state
    # of the working environment

    # logic : just loop through states

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        #print(f"inside play: {env.state}")
        path = [env.state[0]]

        while True:
            # selects the current best action
            state_tuple = tuple(state.tolist())
            action = self.select_action(state_tuple)
            path.append(action)
            #print(action)
            new_state, reward, is_done, _ = env.step(action)
            new_state_tuple = tuple(new_state.tolist())

            self.rewards[(state_tuple, action, new_state_tuple)] = reward
            self.transits[(state_tuple, action)] = new_state_tuple
            total_reward += reward
            if is_done:
                #add last step back to origin node
                #im just worried that this approach doesn't really satisfy the principles
                #of RL. the way TSP in or-gym is designed will not close the loop, or at least
                #that's what i gathered

                total_reward += self.env.distance_matrix[path[0], path[-1]]
                path.append(path[0])

                break
            state = new_state
        return total_reward, path


    def value_iteration(self):
        # loop over all states of the environment

        #for state in range(self.env.observation_space.shape):
        for state,count in self.transits.keys():
            #for every state we calculate the values of the states
            #reachable from state, which will give us candidates for the value
            #of the state

            state_values = [self.calc_action_value(state, action) for action in range(self.env.action_space.n)]

            #select min value that's not negative
            min = 100000
            for val in state_values:
                if val<0:
                    continue
                else:
                    if val < min:
                        min = val

            self.values[state] = min

def display_distances(matrix, path):
    print("Distances in path :")
    for _ in range(1,len(path)):
        dist = matrix[path[_]][path[_-1]]
        print(f"{path[_-1]} --> {path[_]} : {dist}")



