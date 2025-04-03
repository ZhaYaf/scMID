def fuse_nodes(similarities):
    threshold = max(similarities.values()) * 0.95
    nodes = set([node for pair in similarities.keys() for node in pair])
    groups = {}
    for node in nodes:
        groups[node] = {node}
    for (i, j), sim in similarities.items():
        if sim >= threshold:
            merged_group = groups[i].union(groups[j])
            for node in merged_group:
                groups[node] = merged_group

    new_nodes = []
    new_similarities = {}
    for group in set(map(tuple, groups.values())):
        new_nodes.append(list(group))
        for other_group in set(map(tuple, groups.values())):
            if group!= other_group:
                total_sim = 0
                count = 0
                for node1 in group:
                    for node2 in other_group:
                        if (node1, node2) in similarities:
                            total_sim += similarities[(node1, node2)]
                            count += 1
                if count > 0:
                    new_similarities[tuple(group), tuple(other_group)] = round(total_sim / count, 2)

    return new_nodes, new_similarities
