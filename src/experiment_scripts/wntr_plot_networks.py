import wntr
import networkx as nx
from os import path
import matplotlib.pyplot as plt

DEFAULT_NODE_SIZE = 50

def plot_network(network_name, output_file=None):
    data_dir = '../../Data'
    files_for_networks = {
        'Net1': path.join(data_dir, 'Net1', 'net1.inp'),
        'Anytown': path.join(data_dir, 'Anytown', 'anytown.inp'),
        'Anytown_Modified': path.join(data_dir, 'Anytown', 'ATM.inp')
    }
    allowed_networks = files_for_networks.keys()
    if network_name not in allowed_networks:
        raise ValueError(
            f'network_name must be one of {allowed_networks} '
            f'but was {network_name}'
        )
    inp_file = files_for_networks[network_name]
    wn = wntr.network.WaterNetworkModel(inp_file)
    tanks = wn.tank_name_list
    reservoirs = wn.reservoir_name_list
    pos = wn.query_node_attribute('coordinates')
    G = wn.get_graph().to_undirected()
    nx.draw_networkx_nodes(
        G, pos, tanks,
        node_size=DEFAULT_NODE_SIZE,
        node_shape='s',
        node_color='blue',
        label='tanks'
    )
    reservoir_symbols = {
        'Anytown': '<',
        'Anytown_Modified': '<',
        'Net1': '>'
    }
    nx.draw_networkx_nodes(
        G, pos, reservoirs,
        node_size=DEFAULT_NODE_SIZE,
        node_shape=reservoir_symbols[network_name],
        node_color='green',
        label='reservoirs'
    )
    other_nodes = [
        node for node in wn.node_name_list
        if node not in tanks and node not in reservoirs
    ]
    nx.draw_networkx_nodes(
        G, pos, other_nodes,
        node_size=DEFAULT_NODE_SIZE,
        node_color='black'
    )
    pumps = wn.pump_name_list
    edges_by_name = {k: (u,v) for (u,v,k) in G.edges}
    if 'Anytown' in network_name:
        # Plotting pumps is done in this ugly way to make all pumps visible
        nx.draw_networkx_edges(
            G, pos, edgelist=[edges_by_name[pumps[0]]],
            edge_color='red',
            connectionstyle='arc3, rad = -0.5',
            arrows=True
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=[edges_by_name[pumps[1]]],
            edge_color='red',
            connectionstyle='arc3, rad = 0.5',
            arrows=True
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=[edges_by_name[pumps[2]]],
            edge_color='red', label='pumps'
        )
    else:
        # Plot pumps normally
        nx.draw_networkx_edges(
            G, pos, edgelist=[edges_by_name[pump] for pump in pumps],
            edge_color='red', label='pumps'
        )
    other_edges = [edge for edge in wn.link_name_list if edge not in pumps]
    nx.draw_networkx_edges(
        G, pos, edgelist=[edges_by_name[other_edge] for other_edge in other_edges]
    )
    plt.axis('off')
    plt.legend()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)

if __name__=='__main__':
    network_name = 'Net1'
    output_file = '../../Presentation/Figures/net1.png'
    plot_network(network_name, output_file=output_file)

