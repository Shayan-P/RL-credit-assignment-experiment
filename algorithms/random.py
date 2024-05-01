@dataclass
class ExploreFirstAgent:
    num_actions: int
    max_explore: int

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.action_counts = np.zeros(self.num_actions, dtype=int) # action counts n(a)
        self.Q = np.zeros(self.num_actions, dtype=float) # action value Q(a)

    def update_Q(self, action, reward):
        # Update Q action-value given (action, reward)
        # HINT: Keep track of how good each arm is
        #### TODO: update Q value [5pts] ####
        self.Q[action] = (self.Q[action] * self.action_counts[action] + reward) / (self.action_counts[action] + 1)
        self.action_counts[action] += 1


    def get_action(self):
        self.t += 1
        #### TODO: get action [5pts] ####
        if self.t <= self.max_explore:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q)
        ##################################

        return selected_action
