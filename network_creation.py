import pymongo
import igraph

# set up the connection to the db
client = pymongo.MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes  # get the reference to the collection
zipcodes = db.zipcodes


def time_window_graph(nodes, node_attributes, window):
    # nodes : list<list<crime>>
    # node_attrs : list<map<attribute,value>>
    # window : timedelta

    g = igraph.Graph()
    for attr in node_attributes:
        string_attrs = {str(k): v for k, v in attr.iteritems()}
        g.add_vertex(**string_attrs)
    
    print('added {0} nodes'.format(g.vcount()))

    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            print('{0},{1}'.format(i, j))
            # add an edge every time there is a crime within the window

            # look at each crime in this list
            a = nodes[i]
            an = len(a)
            ai = 0
            # move the window over this list and add edges for matches to ai
            b = nodes[j]
            bn = len(b)

            while ai < an:
                # initialize the window markers
                end = 0
                start = 0

                # move the end marker forward
                while end < bn and within_window(a[ai]['date'],
                                                 b[end]['date'],
                                                 window):
                    end += 1

                # move the start marker forward
                while start < end and not within_window(a[ai]['date'],
                                                        b[start]['date'],
                                                        window):
                    start += 1

                # add an edge for every match
                for _ in range(start, end):
                    g.add_edge(i, j, weight=1)

                ai += 1

    # merge the multiedges to get weighted edges
    g.simplify(combine_edges=sum)
    return g


def time_window_crime_graph(time_window, zip_limit=None, node_limit=None):
    # time_window : timedelta
    # zip_limit : list<string>
    # node_limit : int

    # convert a none node_limit to 0
    if node_limit is None:
        node_limit = 0
    # convert single zips to a single element list
    if type(zip_limit) is str:
        zip_limit = [zip_limit]

    # grab all the crimes within the given limits
    if zip_limit is not None:
        all_crimes = crimes.find({'zipcode': {'$in': zip_limit}},
                                 limit=node_limit).sort('date')
    else:
        all_crimes = crimes.find(limit=node_limit).sort('date')

    nodes = [[c] for c in all_crimes]

    return time_window_graph(nodes, all_crimes.rewind(), time_window)


def time_window_zip_graph(time_window, node_zips, node_limit=None):
    attributes = [zipcodes.find({'zip': code})[0] for code in node_zips]
    attributes = [{str(k): v for k, v in d.iteritems()} for d in attributes]
    return time_window_area_graph(time_window,
                                  [[z] for z in node_zips],
                                  attributes,
                                  node_limit)


def time_window_city_graph(time_window, city_names, node_limit=None):
    cities = []
    attributes = []
    for name in city_names:
        # get all the zipcodes in the given city
        zs = [z for z in zipcodes.find({'city': name},
                                       {'city': 1, 'zip': 1, 'longitude': 1, 'latitude': 1})]
        # find the average (lat,lon)
        lat = reduce(lambda s, z: s+float(z['latitude']), zs, 0.0) / len(zs)
        lon = reduce(lambda s, z: s+float(z['longitude']), zs, 0.0) / len(zs)
        zips = [d['zip'] for d in zs]
        cities.append(zips)
        attributes.append({'latitude': lat, 'longitude': lon, 'name': name})

    return time_window_area_graph(time_window, cities, attributes, node_limit)


def time_window_area_graph(time_window, areas, node_attributes, node_limit=None):
    # time_window : timedelta
    # areas : list<list<string>>
    # node_attributes : list<map<string,val>>
    # node_limit : int

    # change a None node limit to 0
    if node_limit is None:
        node_limit = 0

    # get a list of crimes for each area
    nodes = []
    import time
    for area in areas:
        start = time.time()
        # append the crimes for this area
        in_area = crimes.find({'zipcode': {'$in': area}}, {'date': 1, 'zipcode': 1}, limit=node_limit).sort('date')
        # store as a list, not as a cursor
        in_area = [i for i in in_area]

        print('{0} seconds;\t{1} zip codes;\t{2} crimes'.format(time.time() - start, len(area), len(in_area)))
        #in_area = sorted(d, key=lambda x: x['date'])
        #print time.time() - start
        # if node_limit is not None:
        #     in_area = in_area[0:node_limit]  # grab the first bit of the list
        nodes.append(in_area)

    # construct and return the graph
    return time_window_graph(nodes, node_attributes, time_window)


def within_window(time_a, time_b, window):
    """ True if time a is within window of time b, False otherwise."""
    return time_b - time_a <= window >= time_a - time_b
