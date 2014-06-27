__author__ = 'Tobin Yehle'

import json
import logging.config
from community_detection import get_communities
import natsort
import glob
import igraph
from save_networks import get_crime_name


_centrality = {'betweenness': lambda g, node_zipcode: g.betweenness(g.vs.select(zipcode_eq=node_zipcode)[0]),
              'eigenvector': lambda g, node_zipcode: g.eigenvector_centrality(directed = False)[g.vs.select(zipcode_eq=node_zipcode)[0].index],
              'closeness': lambda g, node_zipcode: g.closeness(g.vs.select(zipcode_eq=node_zipcode)[0]),
              'degree': lambda g, node_zipcode: g.degree(g.vs.select(zipcode_eq=node_zipcode)[0])}
logger = logging.getLogger(__name__)
# OUT OF DATE:
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

def get_dynamic_modularity(path, filename, algorithm):
    mod_list = []
    # file_list = os.listdir('{}/networks'.format(path))
    # os.chdir('{}/networks'.format(path))
    file_list = natsort.natsorted(glob.glob('{}/networks/{}*.graphml'.format(path, filename)))
    for f in file_list:
        # if filename in f:
        g = igraph.Graph.Read_GraphML(f)
        #'{}/networks/{}'.format(path, f)
        clust = get_communities(g, 1, path, filename, algorithm=algorithm)
        try:
            #mod_list.append(g.modularity(clust[0], weights = [int(w) if w >0 else 0 for w in g.es['weight']]))
            mod_list.append(str(type(clust[0])))
        except TypeError:
            mod_list.append(0.0)
            print 'IndexError with graph of {} nodes'.format(g.vcount())
    return mod_list


def get_dynamic_node_centrality(path, filename, node_zipcode, measure):
    c_list= []
    # file_list = os.listdir('{}/networks'.format(path))
    # os.chdir('{}/networks'.format(path))
    file_list = natsort.natsorted(glob.glob('{}/networks/{}*.graphml'.format(path, filename)))
    for f in file_list:
        # if filename in f:
        g = igraph.Graph.Read_GraphML(f)
        # '{}/networks/{}'.format(path, f)
        try:
            c_list.append(_centrality[measure](g, node_zipcode))
        except IndexError:
            c_list.append(0.0)
    return c_list


if __name__ == '__main__':
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    logging.basicConfig(level=logging.DEBUG)

    # cities = json.load(open('cities.json'), 'r')
    # areas = ['baltimore', 'los_angeles', 'miami']
    # crime_types = [None, ['Theft'], ['Burglary'], ['Assault']]
    # distances = [0.1, 0.8, 1.6, 2.4, 3.2]
    # delta_name = ['week', 'month', 'year']

    cities = json.load(open('cities.json', 'r'))
    areas = ['miami']
    crime_types = [['all']]
    distances = [1.6]
    delta_name = ['month']


    # Iterate through variables of interest
    for delta in delta_name:
        for dist in distances:
            for crime in crime_types:
                for area in areas:

                    path = 'data/{}/{}/distance/{}/zip'.format(area, get_crime_name(crime), dist)

                    # # Generate modularity measures for dynamic graphs sequence 
                    # modu = dict()
                    # for a in ['multilevel', 'label_propagation', 'fast_greedy']:
                    #     modu[a] = get_dynamic_modularity(path, '{}_{}'.format(delta, '2010'), a)
                    # # Save modu dictionary as json in path/communities
                    # json.dump(modu, open('{}/communities/{}/modularity_{}_2010.json'.format(path, a, delta), 'w'))

                    # Generate centrality measures for dynamic graphs sequence
                    for b in ['betweenness', 'eigenvector', 'closeness', 'degree']:
                        centrality = dict()
                        for z in cities[area]:
                            centrality[z] = get_dynamic_node_centrality(path, '{}_{}'.format(delta, '2010'), z, b)
                        # Save centrality for measure b as json in path/centrality
                        json.dump(centrality, open('{}/centrality/{}_{}_2010.json'.format(path, b, delta), 'w'))

