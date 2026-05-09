
import json
import numpy as np
import os
import networkx as nx
import time
from collections import deque

def calculate_scoap(G, num_nodes, type_map_rev):
    cc0 = {i: 1 for i in range(num_nodes)}
    cc1 = {i: 1 for i in range(num_nodes)}
    co = {i: 0 for i in range(num_nodes)}
    
    # Break cycles at DFFs for topo sort
    G_dag = G.copy()
    for n, data in G_dag.nodes(data=True):
        if 'DFF' in type_map_rev.get(n, ''):
            # Remove outgoing edges to break sequential loops
            out_edges = list(G_dag.out_edges(n))
            G_dag.remove_edges_from(out_edges)
            
    try:
        topo_order = list(nx.topological_sort(G_dag))
        # Controllability
        for n in topo_order:
            g_type = type_map_rev.get(n, "OTHER")
            preds = list(G.predecessors(n))
            if not preds: continue
            p0 = [cc0[p] for p in preds]; p1 = [cc1[p] for p in preds]
            if "AND" in g_type:
                cc0[n] = min(p0) + 1; cc1[n] = sum(p1) + 1
            elif "OR" in g_type:
                cc0[n] = sum(p0) + 1; cc1[n] = min(p1) + 1
            elif "NOT" in g_type:
                cc0[n] = cc1[preds[0]] + 1; cc1[n] = cc0[preds[0]] + 1
            else:
                cc0[n] = min(p0) + 1; cc1[n] = min(p1) + 1
        # Observability
        for n in reversed(topo_order):
            succs = list(G.successors(n))
            if not succs: continue
            co[n] = min([co[s] for s in succs]) + 1
    except: pass
    return cc0, cc1, co

def get_limited_cone_size(G, node, depth_limit=5, reverse=False):
    count = 0
    visited = {node}
    queue = deque([(node, 0)])
    while queue:
        u, d = queue.popleft()
        if d >= depth_limit: continue
        neighbors = G.predecessors(u) if reverse else G.successors(u)
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                count += 1
                queue.append((v, d + 1))
    return count

def parse_netlist_to_gnn_advanced(json_path, output_dir):
    start_total = time.time()
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    module = data['modules'][list(data['modules'].keys())[0]]
    cells = module['cells']; ports = module['ports']
    nodes = sorted(cells.keys()); node_idx = {name: i for i, name in enumerate(nodes)}
    num_nodes = len(nodes)
    
    features = np.zeros((num_nodes, 40))
    # Improved Type Mapping for common tech libraries
    type_map = {
        'AND': 0, 'OR': 1, 'NOT': 2, 'INV': 2, 'NAND': 3, 'NOR': 4, 'XOR': 5, 'XNOR': 5, 'DFF': 6, 'MUX': 7
    }
    type_map_rev = {}

    G = nx.DiGraph()
    for i in range(num_nodes): G.add_node(i)

    net_drivers = {}; net_receivers = {}
    primary_inputs = set(); primary_outputs = set()
    for p_name, port in ports.items():
        bits = port['bits']
        if port['direction'] == 'input': primary_inputs.update(bits)
        else: primary_outputs.update(bits)

    for name, cell in cells.items():
        idx = node_idx[name]
        c_type = cell['type'].upper(); type_map_rev[idx] = c_type
        
        # Heuristic gate type mapping
        tid = 7 # Default to MUX/OTHER
        for key, val in type_map.items():
            if key in c_type:
                tid = val
                break
        
        features[idx, tid] = 1 # 0-7: Type
        features[idx, 8] = 1 if 'DFF' in c_type else 0 # 8: Seq Flag
        
        port_dirs = cell.get('port_directions', {})
        for p, nets in cell['connections'].items():
            # Infer direction if not provided
            if p in port_dirs:
                direction = port_dirs[p]
            else:
                # Heuristic: ports starting with Y, Q, O, Z or containing OUT are outputs
                if any(p.upper().startswith(x) for x in ['Y', 'Q', 'O', 'Z']) or 'OUT' in p.upper():
                    direction = 'output'
                else:
                    direction = 'input'
            
            for net in nets:
                if direction == 'input': net_receivers.setdefault(net, []).append(idx)
                else: net_drivers.setdefault(net, []).append(idx)

    pi_nodes = set(); po_nodes = set()
    print(f"  Primary Inputs: {len(primary_inputs)} bits")
    for net, drvs in net_drivers.items():
        recs = net_receivers.get(net, [])
        for d in drvs:
            for r in recs:
                if d != r: G.add_edge(d, r)
    
    for net in primary_inputs:
        recs = net_receivers.get(net, [])
        pi_nodes.update(recs)
    for net in primary_outputs:
        drvs = net_drivers.get(net, [])
        po_nodes.update(drvs)
        
    print(f"  Detected PI Nodes: {len(pi_nodes)}")
    print(f"  Detected PO Nodes: {len(po_nodes)}")

    # Linear-Time Graph Metrics
    cc0, cc1, co = calculate_scoap(G, num_nodes, type_map_rev)
    
    # Distance to PI/PO (BFS)
    dist_pi = {n: 1e6 for n in range(num_nodes)}
    queue = deque([(n, 0) for n in pi_nodes])
    for n in pi_nodes: dist_pi[n] = 0
    while queue:
        u, d = queue.popleft()
        if d >= dist_pi[u]: # Safety: only propagate if we found a shorter path
            for v in G.successors(u):
                if dist_pi[v] == 1e6:
                    dist_pi[v] = d + 1
                    queue.append((v, d + 1))
            
    dist_po = {n: 1e6 for n in range(num_nodes)}
    queue = deque([(n, 0) for n in po_nodes])
    for n in po_nodes: dist_po[n] = 0
    while queue:
        u, d = queue.popleft()
        if d >= dist_po[u]:
            for v in G.predecessors(u):
                if dist_po[v] == 1e6:
                    dist_po[v] = d + 1
                    queue.append((v, d + 1))

    # Fast Topo Depths
    depths = {n: 0 for n in range(num_nodes)}
    rev_depths = {n: 0 for n in range(num_nodes)}
    try:
        topo = list(nx.topological_sort(nx.DiGraph(G.edges()))) # Simple DAG view
        for n in topo:
            for p in G.predecessors(n): depths[n] = max(depths[n], depths[p] + 1)
        for n in reversed(topo):
            for s in G.successors(n): rev_depths[n] = max(rev_depths[n], rev_depths[s] + 1)
    except: pass

    for i in range(num_nodes):
        features[i, 9] = G.in_degree(i) # 9: Fan-in
        features[i, 10] = G.out_degree(i) # 10: Fan-out
        features[i, 11] = depths[i] # 11: Topo Depth
        features[i, 12] = rev_depths[i] # 12: Rev Depth
        features[i, 13] = min(dist_pi[i], 1000) # 13: Dist PI
        features[i, 14] = min(dist_po[i], 1000) # 14: Dist PO
        features[i, 15] = np.log10(cc0[i] + 1) # 15: CC0
        features[i, 16] = np.log10(cc1[i] + 1) # 16: CC1
        features[i, 17] = np.log10(co[i] + 1) # 17: CO
        # 18: Controllability Imbalance
        features[i, 18] = abs(cc0[i] - cc1[i]) / (cc0[i] + cc1[i] + 1)
        # 19: Recursive Fan-in (Limited)
        features[i, 19] = get_limited_cone_size(G, i, depth_limit=5, reverse=True)
        # 20: Recursive Fan-out (Limited)
        features[i, 20] = get_limited_cone_size(G, i, depth_limit=5, reverse=False)
        # 21: Local Branching
        features[i, 21] = G.out_degree(i) / (G.in_degree(i) + 1)
        
        # 22-29: 1-hop Neighbor Type Histogram
        for neighbor in G.neighbors(i):
            ntid = next((v for k, v in type_map.items() if k in type_map_rev.get(neighbor, '')), 7)
            features[i, 22 + ntid] += 1

    edge_index = np.array(list(G.edges)).T if G.edges else np.empty((2,0))
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'features.npy'), features)
    np.save(os.path.join(output_dir, 'edge_index.npy'), edge_index)
    print(f"  Extracted 40-dim rich features for {num_nodes} nodes in {time.time()-start_total:.2f}s")

if __name__ == "__main__":
    import sys
    parse_netlist_to_gnn_advanced(sys.argv[1], sys.argv[2])
