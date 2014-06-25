__author__ = 'Tobin Yehle'

import igraph
import datetime
import json
import network_creation
from windows import crime_window, str2date
import os.path
from math import ceil
import logging.config

logger = logging.getLogger(__name__)


def save_graph(g, file_path):
    path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(path):
        os.makedirs(path)
    g.write_graphml(file_path)


def save_distance_zip_networks(cache=True):
    city_zips = json.load(open('cities.json', 'r'))
    cities = ['los_angeles', 'baltimore']
    distances = [.1, .8, 1.6, 2.4, 3.2]

    # TODO: Make this parallel if the network creation cannot be made parallel

    for city in cities:
        # grab the data we will use
        data = crime_window(zipcodes=city_zips[city],
                            start_date=datetime(2010, 12, 30),
                            end_date=datetime(2011, 1, 1))
        filename = '30dec2010'
        for d in distances:
            print('{} {}'.format(city, d))
            base_path = 'data/{}/distance/{}'.format(city, d)
            crime_path = '{}/crime/networks/{}.graphml'.format(base_path, filename)
            zip_path = '{}/zip/networks/{}.graphml'.format(base_path, filename)
            # if the graph already exists, do not make a new one
            if os.path.exists(zip_path):
                continue
            if not os.path.exists(crime_path):
                # no crime network found
                if cache:
                    # if we want to cache the crime network, make one
                    g = network_creation.distance_crime_graph(data, d)
                    save_graph(g, crime_path)
                    # reduce and save the network
                    network_creation.reduce_to_zip_graph(g)
                    save_graph(g, zip_path)
                else:
                    # do not cache (probably for memory reasons)
                    g = network_creation.distance_zip_graph(data, d)
                    save_graph(g, zip_path)
            else:
                # the crime network already exists, so grab it
                g = igraph.Graph.Read(crime_path)
                # reduce the network
                network_creation.reduce_to_zip_graph(g)
                save_graph(g, zip_path)


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


def save_dynamic_distance_graph(initial, final, delta, area_name, distance, node_type, crime_types=None):
    """ Creates graphs per each unit of delta time given a window of crime.

        A crime window is created using the relevant given paremeters. For each
        increment of delta between time initial and time final, a graph of the 
        crime window for relevant times is saved as a .graphml file using a 
        unique file name.

        Parameters
        ----------
        initial: datetime.datetime
            Initial time of crimes used to retrieve crime window.
        final: datetime.datetime
            Final time of crimes used to retrieve crime window.
        delta: datetime.timedelta
            Time difference of interest.
        area_name:
            String indicating the name of the city find crimes in.
        distance: float
            Maximum distance between linked crimes.
        node_type: String
            What each node in the network represents. Should be one of 'zip' or 'crime'.
        crime_types: list
            An optional additional parameter passed to `crime_window`

        Returns 
        -------
        .graphml
            Multiple .graphml files with unique names indicative of the time 
            delta at hand.

        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> initial = datetime(2010, 1, 1)
        >>> final = datetime(2010, 1, 8)
        >>> delta = timedelta(days=1)
        >>> save_dynamic_distance_graph(initial, final, delta, 'baltimore', 1.6, 'zip')
    """
    path = 'data/{}/distance/{}/{}'.format(area_name, distance, node_type)
    zipcodes = json.load(open('cities.json', 'r'))[area_name]
    # Create crime window
    w = crime_window(start_date=initial, end_date=final, zipcodes=zipcodes, crime_types=crime_types)
    # Calculte number of increments
    increment = int(ceil((final - initial).total_seconds()/delta.total_seconds()))
    t = initial
    # Create graph for each increment
    for c in range(increment):
        logger.info('Creating graph {} of {}'.format(c, increment))
        cur_t = t + delta
        # Filter relevant crimes
        c_relevant = []
        for crime in w:
            d = str2date(crime['date'])
            if t <= d < cur_t:
                c_relevant.append(crime)
        g = network_creation.distance_graph(c_relevant, distance, node_type)
        g.vs['y'] = [-x for x in g.vs['latitude']]
        g.vs['x'] = [x for x in g.vs['longitude']]
        # Size node proportional to node betweenness
        m = max([g.betweenness(i) for i in g.vs])
        g.vs['size'] = [((g.betweenness(i)/m) * 12) + 5 for i in g.vs]
        save_graph(g, '{}/networks/{}_{}_{}_{}.graphml'.format(path, str(delta.days), initial.strftime('%m%d%Y'), final.strftime('%m%d%Y'), c))
        #igraph.plot(g)
        t = cur_t 


if __name__ == '__main__':
    """ Create dynamic graphs for all of 2010 by days and weeks for baltimore and los angeles for multiple distance networks."""
    from community_detection import get_dynamic_modularity, get_dynamic_node_betweenness
    # Create dictionary
    modularity_dict = dict()
    centrality_dict = dict()
    # Set up variables of interest
    cities = json.load(open('cities.json', 'r'))
    i = datetime.datetime(2010, 1, 1)
    f = datetime.datetime(2011, 1, 1)
    delta_day = datetime.timedelta(days = 1)
    delta_week = datetime.timedelta(days = 7)
    dist1 = 3.2
    dist2 = 0.8
    area1 = 'baltimore'
    area2= 'los_angeles'
    file_name = '{}_{}_'.format(i.strftime('%m%d%Y'), f.strftime('%m%d%Y'))
    
    # Iterate through variables of interest
    for delta in [delta_day, delta_week]:
        for dist in [dist1, dist2]:
            for area in [area1, area2]:
                save_dynamic_distance_graph(i, f, delta, area, dist, 'zip')
                path = 'data/{}/distance/{}/zip'.format(area, str(dist))
                dict_key = '{}-{}-{}'.format(area, dist, delta)
                mod = get_dynamic_modularity(path, '{}_{}'.format(str(delta.days), file_name))
                modularity_dict[dict_key] = mod
                cent = []
                for z in cities[area]:
                    cent.append(get_dynamic_node_betweenness(path, '{}_{}'.format(str(delta.days), file_name), z))
                centrality_dict[dict_key] = cent
    
    # Save as json
    json.dump(open('2010_dynamic_analysis.json'), 'w')
