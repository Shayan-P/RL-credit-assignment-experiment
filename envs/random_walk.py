import networkx as nx
import gymnasium as gym


class RandomWalkEnv(gym.Env):
    def __init__(self):
        self.G = self.create_graph()
        n = self.G.number_of_nodes()
        # todo should we change this to one hot encoding?
        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = gym.spaces.Discrete(n)

        self.starting_vertex = min(self.G.nodes())
        self.ending_vertex = max(self.G.nodes())
        self.reach_the_goal_reward = 10

        ######################################
        self.cur_vertex = self.starting_vertex
        self.step_count = 0
        self.step_count_limit = 0
        self.walk = []
        self.reset()

    @staticmethod
    def create_graph():
        # todo change this graph and randomize it later...
        G = nx.Graph()
        G.add_edge(0, 1, weight=1)
        G.add_edge(1, 2, weight=2)
        G.add_edge(0, 3, weight=1)
        G.add_edge(3, 2, weight=1)
        G.add_edge(1, 3, weight=1)
        G.add_edge(3, 5, weight=1)
        G.add_edge(2, 4, weight=3)
        return G

    def reset(self, seed=None, **kwargs):
        self.cur_vertex = self.starting_vertex
        self.walk = [self.cur_vertex]
        self.step_count = 0
        self.step_count_limit = 8  # todo set this based on the graph
        info = {}
        return self.cur_vertex, info

    def step(self, action):
        observations, rewards, termination, truncation, infos = self.cur_vertex, 0, False, False, {}

        self.step_count += 1

        if action not in self.action_space:
            termination = True
            return observations, rewards, termination, truncation, infos
        if self.step_count >= self.step_count_limit:
            truncation = True

        next_vertex = action
        if next_vertex not in self.G[self.cur_vertex]:
            observations = self.cur_vertex
            rewards = -1  # cost of using an edge that doesn't exist
        else:
            observations = next_vertex
            w = self.G[self.cur_vertex][next_vertex]['weight']
            rewards = -w
            self.cur_vertex = next_vertex
            self.walk.append(self.cur_vertex)
            if next_vertex == self.ending_vertex:
                rewards += self.reach_the_goal_reward
                termination = True
        return observations, rewards, termination, truncation, infos

    def render(self):
        # todo later we can add render_human flag to the environment so that it visualizes as we are training... (in human mode)
        pos = nx.spring_layout(self.G)

        # Plot the graph
        nx.draw(self.G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black', linewidths=1,
                font_size=12, font_weight='bold')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels={(u, v): d['weight'] for u, v, d in self.G.edges(data=True)})

        # plot the nodes in the path traversed
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.walk, node_color='red', node_size=2000, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos,
                               edgelist=[(self.walk[i], self.walk[i + 1]) for i in range(len(self.walk) - 1)],
                               edge_color='red', width=2, alpha=0.8)
