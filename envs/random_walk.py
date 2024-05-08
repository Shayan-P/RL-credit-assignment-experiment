import random

import networkx as nx
import gymnasium as gym


class RandomWalkEnv(gym.Env):
    def __init__(self, num_nodes, weight_max, reach_the_goal_reward, max_episode_length, verbose=False):
        self.verbose = verbose

        self.G = self.create_graph(num_nodes, weight_max)

        n = self.G.number_of_nodes()

        self.action_space = gym.spaces.Discrete(n)
        self.observation_space = gym.spaces.Discrete(n)

        self.starting_vertex = 0
        self.ending_vertex = num_nodes-1
        self.reach_the_goal_reward = reach_the_goal_reward
        self.max_episode_length = max_episode_length

        ######################################
        self.cur_vertex = self.starting_vertex
        self.step_count = 0
        self.step_count_limit = 0
        self.walk = []
        self.reset()

    @staticmethod
    def create_graph(n, weight_max):
        G = nx.Graph()
        # G.add_edge(0, 1, weight=1)
        # G.add_edge(1, 2, weight=2)
        # G.add_edge(0, 3, weight=1)
        # G.add_edge(3, 2, weight=1)
        # G.add_edge(1, 3, weight=1)
        # G.add_edge(3, 5, weight=1)
        # G.add_edge(2, 4, weight=3)
        G.add_nodes_from(range(n))
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    weight = random.randint(1, weight_max)
                    G.add_edge(u, v, weight=weight)
        return G

    def reset(self, seed=None, **kwargs):
        self.cur_vertex = self.starting_vertex
        self.walk = [self.cur_vertex]
        self.step_count = 0
        self.step_count_limit = self.max_episode_length
        info = {}
        if self.verbose:
            print(f"reset. node: {self.cur_vertex}")

        return self.cur_vertex, info

    def step(self, action):
        if self.verbose:
            print(f"node: {self.cur_vertex} action: {action}")

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
