import logging
import numpy as np

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import TimedBehaviour

import networkx as nx
import matplotlib.pyplot as plt

import json
from time import sleep
from random import randint, seed as rand_seed, random


AGENTS_NUM = 10
WEIGHTS_BOUNDARIES = (-100, 100)
# Интервал обновления
DELAY = 2
# Вероятность, что связь есть в i-й момент
CONNECTION_PROBABILITY = 0.7
# Вероятность, что связь будет в принципе 
# (Здесь актуально только для большого количества вершин, потому что у меня граф связный)
LINK_FORM_PROBABILTY = 0.3

# Помехи
INTERFERENCE_STD = 0.1
# INTERFERENCE_STD = None
# Показывать граф или нет
SHOW_GRAPH = True

RAND_SEED = None


# Определяет промежуток, через который будут отправляться сообщения
class SendBehaviour(TimedBehaviour):
    def __init__(self, agent, time):
        super(SendBehaviour, self).__init__(agent, time)

    def on_time(self):
        super(SendBehaviour, self).on_time()
        for neighbor_idx in self.agent.neighbors:
            if random() <= self.agent.neighbors[neighbor_idx]:
                self.agent.send_message(neighbor_idx)
        

class AVGAgent(Agent):
    def __init__(self, aid, idx, num, neighbors, mapping_idx_agent_name, delay=5):
        super(AVGAgent, self).__init__(aid=aid)
        self.aid = aid
        self.idx = idx
        self.num = num
        self.neighbors = neighbors
        self.mapping_idx_agent_name = mapping_idx_agent_name
        self.known_weights = {idx: num}
        self.heard_weights_times = {}
        display_message(self.aid.localname, f'Мой номер: {idx}. Моё число: {num}')

        send_behaviour = SendBehaviour(self, delay)
        self.behaviours.extend([send_behaviour, 
                                ])

    def send_message(self, neighbor_idx):
        msg = ACLMessage()
        msg.add_receiver(AID(name=self.mapping_idx_agent_name[neighbor_idx]))
        
        if INTERFERENCE_STD is not None:
            errors =  np.random.normal(0, INTERFERENCE_STD, len(self.known_weights))
            erorrs_weights = {}
            for i, weight in enumerate(self.known_weights):
                erorrs_weights[weight] = self.known_weights[weight] + errors[i]
        else:
            erorrs_weights = self.known_weights.copy()
        msg.set_content(json.dumps(erorrs_weights))
        self.send(msg)

    def react(self, message):
        super(AVGAgent, self).react(message)
        if not message.system_message:
            try:
                # Иначе ключи будут строками
                got_weights = {int(key): value for key, value in json.loads(message.content).items()}
                if self.idx in got_weights.keys():
                    got_weights.pop(self.idx)
                # Обновляем avg
                for idx in got_weights:
                    if idx not in self.heard_weights_times:
                        self.heard_weights_times[idx] = 1
                        self.known_weights[idx] = got_weights[idx]
                    else:
                        self.known_weights[idx] = (self.heard_weights_times[idx] * self.known_weights[idx]  + got_weights[idx]) / \
                            (self.heard_weights_times[idx] + 1)
                        self.heard_weights_times[idx] += 1
            except (json.decoder.JSONDecodeError, AttributeError, UnicodeDecodeError) as decoder_ex:
                logging.debug('Не может быть декодировано: '+ str(message.content))

            # Считаем среднее исходя из 
            display_message(self.aid.localname,
                             f'Посчитанное среднее: {sum(self.known_weights.values()) / len(self.known_weights)}.\n'
                             f" Я знаю про следующие вершины: {self.known_weights}"
                             )
            
def make_graph(indexes, numbers, link_form_probabilty=1, show_plot=False):
    graph = nx.erdos_renyi_graph(len(indexes), link_form_probabilty)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(len(indexes), link_form_probabilty)

    for node in graph.nodes():
        graph.nodes[node]['weight'] = numbers[node]

    mapping = {i: indexes[i] for i in range(len(indexes))}
    graph = nx.relabel_nodes(graph, mapping)
    

    if show_plot:
            fig = plt.figure()
            fig.canvas.manager.set_window_title("Граф связей")

            pos = nx.spring_layout(graph)
            nx.draw(graph, pos, with_labels=False, node_color='lightblue')
            labels = {node: f"{node}\n(Вес {data['weight']})" for node, data in graph.nodes(data=True)}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10)

            plt.show()
    return graph
    
# Аналог симметричной матрицы смежности в словаре
def make_probability_dict(graph, indexes, connection_probability=1):
    connection_probability_dict = {}
    for edge_1, edge_2 in graph.edges():
        connection_probability_dict[(edge_2, edge_1)] = \
            connection_probability_dict[(edge_1, edge_2)] = connection_probability
    return connection_probability_dict


if __name__ == '__main__':
    if RAND_SEED is not None:
        rand_seed(RAND_SEED)

    agents_num = AGENTS_NUM
    indexes = tuple(range(1, 1 + agents_num))
    ports = tuple(range(10000, 10000 + agents_num))
    weights = [randint(WEIGHTS_BOUNDARIES[0], WEIGHTS_BOUNDARIES[1]) for _ in range(agents_num)]
    # Символы для подсветки красным
    print(f"\033[91mНастоящее среднее: {sum(weights) / len(weights)}\033[0m")

    graph = make_graph(indexes, weights, link_form_probabilty=LINK_FORM_PROBABILTY, show_plot=SHOW_GRAPH)
    connection_probability_dict = make_probability_dict(
        graph,
        indexes,
        connection_probability=CONNECTION_PROBABILITY,
        )

    agent_names = {idx: 'agent_avg_{}@localhost:{}'.format(idx, port) for idx, port in zip(indexes, ports)}
    agents = []

    for idx, port, num in zip(indexes, ports, weights):
        agent_name = agent_names[idx]
        
        # Находим для каждого агента вероятность связи с соседями (про тех, что с вероятностью 0, агент не будет знать)
        neighbors_probabilities = {}
        for j_idx in indexes:
            if j_idx != idx and (idx, j_idx) in connection_probability_dict:
                neighbors_probabilities[j_idx] = connection_probability_dict[(idx, j_idx)] 

        # Здесь сохраняем имена для каждого соседа
        neighbors_idx_agent_name_mapping = {j_idx: agent_names[j_idx] 
                                            for j_idx in neighbors_probabilities.keys()}
        agents.append(
            AVGAgent(AID(name=agent_name),
                      idx,  num, neighbors_probabilities, neighbors_idx_agent_name_mapping,
                      delay=DELAY))

    start_loop(agents)
