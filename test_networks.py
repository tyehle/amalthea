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
        zip_data = zipcodes.find_one({'zip': all_crimes[i]['zipcode']}, limit=1)
        try:
            cities |= {(str(zip_data['city']), str(zip_data['state']))}
        except:
            print('No entries in zip database for {0}'.format(all_crimes[i]['zipcode']))
        finally:
            i += 1


    g = time_window_city_graph(window, cities, node_limit=1000)
    g.vs['label'] = g.vs['city']
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]
    igraph.plot(g, edge_width=[math.log(w) for w in g.es['weight']])


def test_sequential_crime_graph():
    g = sequential_crime_graph(zip_limit='33611', node_limit=30)
    g.vs['label'] = g.vs['date']
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]
    igraph.plot(g)


def test_sequetial_zip_graph():
    # find some zip codes
    total = 10
    zips = set()
    i = 0
    all_crimes = crimes.find()
    while len(zips) < total:
        zips |= {str(all_crimes[i]['zipcode'])}
        i += 1
    g = sequential_zip_graph(list(zips), crime_limit=3000)
    igraph.plot(g, edge_width=[math.log(w) for w in g.es['weight']])

# find some city names
window = timedelta(hours=8)
total = 60
cities = set()
i = 0
all_crimes = crimes.find()
while len(cities) < total:
    zip_data = zipcodes.find_one({'zip': all_crimes[i]['zipcode']}, limit=1)
    try:
        cities |= {(str(zip_data['city']), str(zip_data['state']))}
    except:
        print('No entries in zip database for {0}'.format(all_crimes[i]['zipcode']))
    finally:
        i += 1

g = sequential_city_graph(cities, crime_limit=300000)
g.vs['label'] = g.vs['city']
g.vs['x'] = [float(x) for x in g.vs['longitude']]
g.vs['y'] = [-float(y) for y in g.vs['latitude']]
igraph.plot(g, edge_width=[math.log(w) for w in g.es['weight']])