from network_creation import *
from datetime import timedelta
import igraph
import math


def test_time_crime_graph():
    window = timedelta(hours=8)
    code = '33611'
    nodes = 10
    g = time_window_crime_graph(window, zip_limit=code, node_limit=nodes)

    g.vs['label'] = [d.time() for d in g.vs['date']]


def test_time_zip_graph():
    # find some zip codes
    window = timedelta(hours=8)
    total = 100
    zips = set()
    i = 0
    all_crimes = crimes.find()
    while len(zips) < total:
        zips |= {str(all_crimes[i]['zipcode'])}
        i += 1

    g = time_window_zip_graph(window, zips, node_limit=1000)
    g.vs['label'] = g.vs['zip']
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [float(y) for y in g.vs['latitude']]
    igraph.plot(g, edge_width=[math.log(w) for w in g.es['weight']])


def test_time_city_graph():
    # find some city names
    window = timedelta(hours=8)
    total = 100
    cities = set()
    i = 0
    all_crimes = crimes.find()
    while len(cities) < total:
        zip_data = zipcodes.find({'zip': all_crimes[i]['zipcode']}, limit=1)
        try:
            cities |= {str(zip_data[0]['city'])}
        except:
            print('{0} entries in zip database for {1}'.format(zip_data.count(), all_crimes[i]['zipcode']))
        finally:
            i += 1


    g = time_window_city_graph(window, cities, node_limit=1000)
    g.vs['label'] = g.vs['name']
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]
    igraph.plot(g, edge_width=[math.log(w) for w in g.es['weight']])


test_time_city_graph()