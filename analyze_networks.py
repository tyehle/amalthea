__author__ = 'Tobin Yehle'

import math
import igraph
from datetime import datetime
import json
import network_creation
from windows import crime_window
import os.path
from math import ceil
import logging.config

logger = logging.getLogger(__name__)

def save_graph(g, file_path):
    dir = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(dir):
        os.makedirs(dir)
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


def get_dynamic_graph(graph_creator, delta, initial, final, path, zipcodes = None):
    """ Creates graphs per each unit of delta time given a window of crime.

        A crime window is created using the relevant given paremeters. For each
        increment of delta between time initial and time final, a graph of the 
        crime window for relevant times is saved as a .graphml file using a 
        unique file name.

        Parameters
        ----------
        graph_creator: function
            Function that returns a igraph.Graph() with the parameters of interest specified (other than window).
        delta: datetime.timedelta
            Time difference of interest.
        initial: datetime.datetime
            Initial time of crimes used to retrieve crime window.
        final: datetime.datetime
            Final time of crimes used to retrieve crime window.
        path:
            String indicating where the dynamic graphs should be saved.
        zipcodes: list
            List of zip code strings.

        Returns 
        -------
        .graphml
            Multiple .graphml files with unique names indicative of the time 
            delta at hand.

        Examples
        --------
        >>> initial = datetime.datetime(2010, 1, 1)
        >>> final = datetime.datetime(2010, 1, 8)
        >>> graph_creator = lambda x: network_creation.distance_zip_graph(
            x, 1.6) 
        >>> delta = datetime.timedelta(days = 1)
        >>> import json
        >>> cities = json.load(open('cities.json'))
        >>> path = 'amalthea/data/baltimore/distance/1.6/zip/networks'
        >>> create_dynamic_graphs(graph_creator, delta, initial, final, zipcodes = cities['baltimore'], file_name = 'path' + '/')
    """
    # Create crime window
    w = crime_window(start_date = initial, end_date = final, zipcodes = zipcodes)
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
            d = datetime.datetime.strptime(crime['date'], "%Y-%m-%d %H:%M:%S")
            if t <= d < cur_t:
                c_relevant.append(crime)
        g = graph_creator(c_relevant)
        g.vs['y'] = [-x for x in g.vs['latitude']]
        g.vs['x'] = [x for x in g.vs['longitude']]
        g.vs['size'] = g.vs['description']/10.0
        save_graph(g, '{}/{}_{}'.format(path, t.date(), t.time()))
        t = cur_t 


if __name__ == '__main__':
    # analyze_graphs()
    save_distance_zip_networks()
