import os
import graphviz
import neat


def draw_net(
    config,
    genome,
    filename=None,
    view=False,
    node_names=None,
    show_disabled=True,
    prune_unused=False,
    node_colors=None,
    fmt="png",
):
    """Draws a neural network with Graphviz."""
    from graphviz import Digraph

    if graphviz is None:
        raise ImportError("This function requires graphviz")

    if filename is None:
        filename = "net"

    dot = Digraph(format=fmt, node_attr={"shape": "circle", "fontsize": "9", "height": "0.2", "width": "0.2"})

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    for k in inputs:
        name = node_names.get(k, str(k)) if node_names else str(k)
        dot.node(name, _attributes={"style": "filled", "fillcolor": "lightgray"})

    for k in outputs:
        name = node_names.get(k, str(k)) if node_names else str(k)
        dot.node(name, _attributes={"style": "filled", "fillcolor": "lightblue"})

    used_nodes = set(genome.nodes.keys()) if not prune_unused else set()
    if prune_unused:
        connections = [cg for cg in genome.connections.values() if cg.enabled or show_disabled]
        for conn in connections:
            used_nodes.add(conn.key[0])
            used_nodes.add(conn.key[1])

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue
        name = node_names.get(n, str(n)) if node_names else str(n)
        color = "white"
        if node_colors and n in node_colors:
            color = node_colors[n]
        dot.node(name, _attributes={"style": "filled", "fillcolor": color})

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input, output = cg.key
            if prune_unused and (input not in used_nodes or output not in used_nodes):
                continue
            a = node_names.get(input, str(input)) if node_names else str(input)
            b = node_names.get(output, str(output)) if node_names else str(output)
            style = "solid" if cg.enabled else "dotted"
            color = "green" if cg.weight > 0 else "red"
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={"style": style, "color": color, "penwidth": width})

    dot.render(filename, view=view)
