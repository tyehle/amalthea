__author__ = 'Tobin Yehle'

import matplotlib
matplotlib.use('Agg')  # this fixes issues when executing over ssh
import json
import logging.config
from community_detection import get_communities
import natsort
import glob
import igraph
from save_networks import get_crime_name
import matplotlib.pyplot as plt
import cairo
from igraph.drawing.text import TextDrawer
from scipy.spatial.distance import pdist, squareform
import numpy as np
from pymongo import MongoClient
from shapely.geometry import asShape
from community_detection import ensure_folder
from scipy.stats.stats import pearsonr  

_client = MongoClient('163.118.78.22', 27017)
_db = _client['crimes_test']
_geometry = _db.geometry

_centrality = {'betweenness': lambda g, node_zipcode: g.betweenness(g.vs.select(zipcode_eq=node_zipcode)[0], weights = 'weight'),
              'eigenvector': lambda g, node_zipcode: g.eigenvector_centrality(directed = False, weights = 'weight')[g.vs.select(zipcode_eq=node_zipcode)[0].index],
              'closeness': lambda g, node_zipcode: g.closeness(g.vs.select(zipcode_eq=node_zipcode)[0], weights = 'weight'),
              'degree': lambda g, node_zipcode: sum(g.es.select(_source=g.vs.select(zipcode_eq=node_zipcode)[0].index)['weight']) }
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
    file_list = natsort.natsorted(glob.glob('{}/networks/{}*.graphml'.format(path, filename)), number_type=None)
    for f in file_list:
        g = igraph.Graph.Read_GraphML(f)
        clust = get_communities(g, 1, path, filename, algorithm=algorithm)
        try:
            mod_list.append(g.modularity(clust[0], weights = 'weight'))
        except TypeError:
            mod_list.append(0.0)
            print 'IndexError with graph of {} nodes'.format(g.vcount())
    return mod_list


def get_dynamic_node_centrality(path, filename, node_zipcode, measure):
    c_list= []
    file_list = natsort.natsorted(glob.glob('{}/networks/{}*.graphml'.format(path, filename)), number_type=None)
    for f in file_list:
        try:
            g = igraph.Graph.Read_GraphML(f)
        except igraph._igraph.InternalError:
            c_list.append(0.0)
        else:
            try:
                c_list.append(_centrality[measure](g, node_zipcode))
            except (KeyError, IndexError, igraph._igraph.InternalError):
                c_list.append(0.0)
    return c_list


def plot_edge_weight(crime_types, distance, year):
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    
    # Create an axes instance
    ax = fig.add_subplot(111)

    # Load indicated graphs
    data = []
    lab = []
    # cities = ['baltimore', 'los_angeles', 'miami']
    cities = ['baltimore']
    for i in range(12):
        for city in cities:
            mon = str(int(i) + 1)
            if len(mon) == 1:
                mon = '0' + mon
            date = '{}-{}-01'.format(year, mon)
            g = igraph.Graph.Read_GraphML('data/{}/{}/distance/{}/zip/networks/{}_{}.graphml'.format(city, get_crime_name(crime_types), distance, 'month', date))
            #tot = sum(g.vs['description']) * (sum(g.vs['description']) - 1) * 0.5
            #data.append([x/tot for x in g.es['weight']])
            data.append(g.es['weight'])
            lab.append(city[0])

    # Create the boxplot
    bp = ax.boxplot(data)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticklabels(lab)

    # Change colors per month
    c = [['pink', 'orange', 'blue'] for i in range(12)]
    # for i, box in enumerate(bp['boxes']):
    #     # change outline color
    #     box.set( color='red', linewidth=2)
    # for i, whisker in enumerate(bp['whiskers']):
    #     whisker.set(color=c[i], linewidth=2)
    # for i, cap in enumerate(bp['caps']):
    #     cap.set(color=c[i], linewidth=2)
    # for median in bp['medians']:
    #     median.set(color='red', linewidth=2)
    # Save the figure
    fig.savefig('fig1.png', bbox_inches='tight')


def plot_four(g1, g2, g3, g4, title):
    """ Plots four graphs on a single plot.
    """
    # Construct plot
    plot = igraph.Plot(title, bbox=(1200, 940), background="white")
    plot.add(g1, bbox=(20, 60, 580, 470))
    plot.add(g2, bbox=(620, 60, 1180, 470))
    plot.add(g3, bbox=(20, 510, 580, 920))
    plot.add(g4, bbox=(620, 510, 1180, 920))

    plot.redraw()
    # Add title
    ctx = cairo.Context(plot.surface)
    ctx.set_font_size(24)
    drawer = TextDrawer(ctx, 'crime count: {}, zip code count: {}'.format(sum([int(i) for i in  g1.vs['description']]), g1.vcount()), halign=TextDrawer.CENTER)
    drawer.draw_at(0, 40, width=1200)
    plot.save()


def mult_graphs(t_list, title):
    """ Plots the same graph indicative of four different centrality measures. 

        Input: list of tuples describing path of graphs of interest.
    """
    for t in t_list:
        g_return = []
        for measure in ['betweenness', 'degree', 'eigenvector',  'closeness']:
            # Load graph
            g = igraph.Graph.Read_GraphML('data/{}/{}/distance/{}/zip/networks/{}_{}.graphml'.format(*t))
            g.vs['x'] = [float(i) for i in g.vs['longitude']]
            g.vs['y'] = [-float(i) for i in g.vs['latitude']]
            g.es['color'] = 'grey'
            if measure == 'betweenness':
                n = (g.vcount() - 1)*(g.vcount() - 2)*0.5
                g.vs['size'] = [(i/n) * 90 for i in g.betweenness()]
                g.vs['color'] = ['blue' if i/n == max([i/n for i in g.betweenness() ]) else 'red' for i in g.betweenness()]
                g.vs['label'] = ['{}, {}'.format(node['zipcode'], node['description']) if node['size'] > 5 else None for node in g.vs]
                g_return.append(g)
            elif measure == 'degree':
                n = float(max(g.degree()))
                g.vs['size'] = [(i/n) * 8.5 for i in g.degree()]
                g.vs['color'] = ['blue' if i/n == max([i/n for i in g.degree()])    else 'red' for i in g.degree()]
                g.vs['label'] = ['{}, {}'.format(node['zipcode'], node['description']) if node['size'] > 8 else None for node in g.vs]
                g_return.append(g)
            elif measure == 'eigenvector':
                g.vs['size'] = [i * 8.5 for i in g.eigenvector_centrality(directed  = False)]
                g.vs['color'] = ['blue' if i == max(g.vs['size']) else 'red' for i  in g.vs['size']]
                g.vs['label'] = ['{}, {}'.format(node['zipcode'], node['description']) if node['size'] > 8 else None for node in g.vs]
                g_return.append(g)
            elif measure == 'closeness':
                n = float(max(g.closeness()))
                g.vs['size'] = [i/n * 8.5 for i in g.closeness()]
                g.vs['color'] = ['blue' if i == max(g.vs['size']) else 'red' for i  in g.vs['size']]
                g.vs['label'] = ['{}, {}'.format(node['zipcode'], node['description']) if node['size'] > 8.45 else None for node in g.vs]
                g_return.append(g)
        # Call plot for the four respective graphs
        plot_four(g_return[0], g_return[1], g_return[2], g_return[3], '{}_{}'.format(t[-1], title))


def centrality_corr(path):
    # Load json file of dynamic centrality
    info = json.load(open('data/{}/{}/distance/{}/zip/centrality/{}_{}_{}.json'.format(*path)))
    # Find correlation matrix
    x = [info[i] for i in sorted(info.keys())]
    y = np.nan_to_num(squareform(pdist(x, 'correlation')))
    # Return correlation of touching zip codes
    geom = []
    for z_i, z in enumerate(sorted(info.keys())):
        geom.append(asShape(_geometry.find_one({'zip': z})['geometry']))
        for i in range(len(geom) - 1):
            if geom[-1].intersects(geom[i]):
                print('correlation between {} and {}: {}'.format(z, sorted(info.keys())[i], y[z_i][i]))
    return y


def centrality_corr_neighbors_single(zipcode, path):
    """ Find the relationship between node of given zipcode at t0 and node's neighbors at t1
    """
    # Load relevant centrality measure for graph of interest
    info = json.load(open('data/{}/{}/distance/{}/zip/centrality/{}_{}_{}.json'.format(*path)))
    zipcode_cent = info[zipcode]
    zipcode_shape = asShape(_geometry.find_one({'zip': zipcode})['geometry'])
    # Delete keys if they do not interest zipcode of interest
    for z in info.keys():
        if not asShape(_geometry.find_one({'zip': z})['geometry']).intersects(zipcode_shape):
            del info[z]
    y = dict()
    for z in sorted(info.keys()):
        zip_mon = []
        for i in range(len(zipcode_cent) - 1):
            zip_mon.append((zipcode_cent[i] - info[z][i + 1])**2)
        y[z] = zip_mon
    return y

def centrality_corr_neighbors_multiple(zipcode, path):
    """ Find the relationship between node of given zipcode at t0 and node's neighbors at t1
    """
    # Load relevant centrality measure for graph of interest
    info = json.load(open('data/{}/{}/distance/{}/zip/centrality/{}_{}_{}.json'.format(*path)))
    zipcode_cent = info[zipcode][:-1]
    zipcode_shape = asShape(_geometry.find_one({'zip': zipcode})['geometry'])
    # Delete keys if they do not interest zipcode of interest
    for z in info.keys():
        if not asShape(_geometry.find_one({'zip': z})['geometry']).intersects(zipcode_shape):
            del info[z]
    y = dict()
    for z in sorted(info.keys()):
        y[str(z)] = pearsonr(zipcode_cent, info[z][1:])[0]
    return y

if __name__ == '__main__':
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    logging.basicConfig(level=logging.DEBUG)

    cities = json.load(open('cities.json', 'r'))
    areas = ['baltimore', 'los_angeles', 'miami']
    crime_types = [None, ['Theft'], ['Burglary'], ['Assault']]
    distances = [0.1, 0.8, 1.6, 2.4, 3.2]
    delta_name = ['week', 'year'] 

for mon in range(12):
    # Iterate through variables of interest
    for delta in delta_name:
        for dist in distances:
            for crime in crime_types:
                for area in areas:
                    for y in ['2008', '2009', '2010']:
                        
                        # Add centrality directory
                        path = 'data/{}/{}/distance/{}/zip'.format(area, get_crime_name(crime), dist)
                        
                        # # Generate modularity measures for dynamic graphs sequence 
                        # modu = dict()
                        # for a in ['multilevel', 'label_propagation', 'fast_greedy']:
                        #     modu[a] = get_dynamic_modularity(path, '{}_{}'.format(delta, '2009'), a)
                        # # Save modu dictionary as json in path/communities
                        # json.dump(modu, open('{}/communities/{}/modularity_{}_2009.json'.format(path, a, delta), 'w'))

                        # Generate centrality measures for dynamic graphs sequence
                        for b in ['betweenness', 'eigenvector', 'closeness', 'degree']:
                            centrality = dict()
                            for z in cities[area]:
                                centrality[z] = get_dynamic_node_centrality(path, '{}_{}'.format(delta, y), z, b)
                            # Save centrality for measure b as json in path/centrality
                            save_path = '{}/centrality/{}_{}_{}.json'.format(path, b, delta, y)
                            ensure_folder(save_path)
                            json.dump(centrality, open(save_path, 'w'))