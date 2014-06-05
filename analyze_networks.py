__author__ = 'Tobin Yehle'

import math
import igraph
from datetime import datetime
import json
import network_creation
from windows import crime_window

def save_networks():
    cities = json.load(open('cities.json', 'r'))
    # grab the data we will use
    data = crime_window(zipcodes=cities['Los Angeles'],
                        start_date=datetime(2010, 12, 1),
                        end_date=datetime(2011, 1, 1))

    distances = [.1, .8, 1.6, 2.4, 3.2]
    for d in distances:
        print('\ndistance = {0}\n'.format(d))
        g = network_creation.distance_zip_graph(data, d)
        g.write_graphml('output/la_zip_distance_{0}.graphml'.format(d))


def analyze_graphs():
    """ Run analysis on saved networks. """
    path = 'output/10k/{0}_{1}_distance_{2}.graphml'
    cities = ['la', 'baltimore']
    types = ['crime', 'zip']
    distances = [.1, .8, 1.6, 2.4, 3.2]

    output_file = open('output/graph_properties.csv', 'w')

    output_file.write('City,Type,Distance,Nodes,Edges,Transitivity,Degree Assortativity,Diameter,Giant Component\n')
    output = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'

    # run the analysis on all the networks we have
    for c in cities:
        for t in types:
            for d in distances:
                f = path.format(c, t, d)
                g = igraph.Graph.Read(f)
                stats = output.format(c,
                                      t,
                                      d,
                                      g.vcount(),
                                      g.ecount(),
                                      g.transitivity_undirected(),
                                      g.assortativity_degree(),
                                      g.diameter(),
                                      g.components().giant().vcount())
                print(stats)
                output_file.write(stats)


if __name__ == '__main__':
    # analyze_graphs()
    save_networks()
