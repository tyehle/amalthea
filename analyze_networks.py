__author__ = 'Tobin Yehle'

import json
import logging.config

logger = logging.getLogger(__name__)


# def analyze_graphs():
#     """ Run analysis on saved networks. """
#     path = 'output/10k/{0}_{1}_distance_{2}.graphml'
#     cities = ['la', 'baltimore']
#     types = ['crime', 'zip']
#     distances = [.1, .8, 1.6, 2.4, 3.2]
#
#     output_file = open('output/graph_properties.csv', 'w')
#
#     output_file.write('City,Type,Distance,Nodes,Edges,Transitivity,Degree Assortativity,Diameter,Giant Component\n')
#     output = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'
#
#     # run the analysis on all the networks we have
#     for c in cities:
#         for t in types:
#             for d in distances:
#                 f = path.format(c, t, d)
#                 g = igraph.Graph.Read(f)
#                 stats = output.format(c,
#                                       t,
#                                       d,
#                                       g.vcount(),
#                                       g.ecount(),
#                                       g.transitivity_undirected(),
#                                       g.assortativity_degree(),
#                                       g.diameter(),
#                                       g.components().giant().vcount())
#                 print(stats)
#                 output_file.write(stats)


if __name__ == '__main__':
    """ Create dynamic graphs for all of 2010 by days and weeks for baltimore
        and los angeles for multiple distance networks."""
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    # logging.basicConfig(level=logging.DEBUG)

    # # Iterate through variables of interest
    # for _delta in [delta_day, delta_week]:
    #     for dist in [dist1, dist2]:
    #         for area in [area1, area2]:
    #             save_dynamic_distance_graph(i, f, _delta, area, dist, 'zip')
    #             ## NO path = 'data/{}/distance/{}/zip'.format(area, str(dist))
    #             dict_key = '{}-{}-{}'.format(area, dist, _delta)
    #             mod = get_dynamic_modularity(path, '{}_{}'.format(str(_delta.days), file_name))
    #             modularity_dict[dict_key] = mod
    #             cent = []
    #             for z in cities[area]:
    #                 cent.append(get_dynamic_node_betweenness(path, '{}_{}'.format(str(_delta.days), file_name), z))
    #             centrality_dict[dict_key] = cent
    #
    # # Save as json
    # json.dump(open('2010_dynamic_analysis.json'), 'w')
