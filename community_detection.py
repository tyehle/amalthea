__author__ = 'Tobin Yehle'

import igraph
import math


def layout_position(g):
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]


def get_bounds(g, max_dimension=1600):
    xs = [float(x) for x in g.vs['longitude']]
    ys = [-float(y) for y in g.vs['latitude']]
    bounds = (0, 0, max(xs) - min(xs), max(ys) - min(ys))
    scale_factor = max_dimension / max(bounds)
    return [int(x*scale_factor) for x in bounds]


def find_clusters(g):
    comms = g.community_fastgreedy(weights='weight').as_clustering()
    return comms

if __name__ == '__main__':
    # grab a graph from a file
    # good: multilevel @ 0.88, leading eigenvector @ 0.85, walktrap @ 0.85
    g = igraph.Graph.Read('output/la_zip_distance_0.8.graphml')
    bounds = get_bounds(g)
    layout_position(g)
    g.vs['size'] = 8
    c = find_clusters(g)
    igraph.plot(c, mark_groups=True, bbox=bounds)
