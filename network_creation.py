import pymongo
import igraph

# set up the connection to the db
client = pymongo.MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes  # get the reference to the collection
zipcodes = db.zipcodes


# TODO: Change this crap to generate nodes based on a data set


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
    for name, state in city_names:
        # get all the zip codes in the given city
        zs = [z for z in zipcodes.find({'city': name, 'state': state})]

        if len(zs) is 0:
            # print an warning and ignore this city
            print('Warning: no zip codes found for \"{0}, {1}\"'.format(name, state))
            continue

        # find the average (lat,lon)
        lat = reduce(lambda s, z: s+float(z['latitude']), zs, 0.0) / len(zs)
        lon = reduce(lambda s, z: s+float(z['longitude']), zs, 0.0) / len(zs)
        zips = [d['zip'] for d in zs]
        cities.append(zips)
        attributes.append({'latitude': lat, 'longitude': lon, 'city': name, 'state': state, 'zips': zips})

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


def sequential_crime_graph(zip_limit=None, node_limit=None):
    """ Returns a graph where each node represents a crime, and each sequential
        crime by date is connected by an edge.
    """

    if node_limit is None:
        node_limit = 0

    # get the list of all crimes within the limits
    if zip_limit is not None:
        if type(zip_limit) is str:
            zip_limit = [zip_limit]
        cs = crimes.find({'zipcode': {'$in': zip_limit}}, limit=node_limit).sort('date')
    else:
        cs = crimes.find(limit=node_limit).sort('date')

    # make a graph
    g = igraph.Graph()

    # iterate through the graph, linking each node to the previous one
    last_vertex = None
    for c in cs:
        g.add_vertex(**{str(k): v for k, v in c.iteritems()})
        if last_vertex is not None:
            g.add_edge(last_vertex, last_vertex+1, weight=1)
            last_vertex += 1
        else:
            last_vertex = 0

    return g


def sequential_zip_graph(node_zips, crime_limit=None):
    """ Generates a graph where each node represents a single zip code. The
        weights of the edges between nodes are proportional to the number of
        sequential crimes between the two nodes.

        The crime limit will restrict the number of crimes the algorithm
        retrieves from the database.
    """

    if crime_limit is None:
        crime_limit = 0

    areas = []
    for z in node_zips:
        # find all the info in the zipcode database
        info = zipcodes.find_one({'zip': z})
        # convert keys to strings
        info = {str(k): v for k, v in info.iteritems()}
        areas.append(info)

    # define the function to check if a crime is in an area
    crime_in_area = lambda c, a: c['zipcode'] == a['zip']

    # get the list of crimes to build the graph with
    all_crimes = crimes.find({'zipcode': {'$in': node_zips}}, limit=crime_limit).sort('date')

    return sequential_area_graph(areas, crime_in_area, all_crimes)


def sequential_city_graph(city_names, crime_limit=None):
    """ Build a graph in which each node represents a city, and the edge
        weights are proportional to number of sequential crimes between two
        cities.

        The crime limit will restrict the number of crimes the algorithm
        retrieves from the database.
    """

    if crime_limit is None:
        crime_limit = 0

    # find all relevant information about each city
    areas = []
    all_zips = []
    for city, state in city_names:
        # find all zip codes in the city
        zips = [z for z in zipcodes.find({'city': city, 'state': state})]

        if len(zips) is 0:
            # print a warning and ignore this city
            print('Warning: no zip codes found for \"{0}, {1}\"'.format(city, state))
            continue

        # find the average (lat,lon)
        lat = reduce(lambda s, z: s+float(z['latitude']), zips, 0.0) / len(zips)
        lon = reduce(lambda s, z: s+float(z['longitude']), zips, 0.0) / len(zips)

        codes = [d['zip'] for d in zips]
        all_zips.extend(codes)

        areas.append({'latitude': lat, 'longitude': lon, 'city': city, 'state': state, 'zips': codes})

    crime_in_area = lambda c, a: c['zipcode'] in a['zips']
    crime_list = crimes.find({'zipcode': {'$in': all_zips}}, limit=crime_limit).sort('date')

    return sequential_area_graph(areas, crime_in_area, crime_list)


def sequential_area_graph(areas, crime_in_area, crime_list):
    """ Generates a graph where each area is a node, and the weight of each
        node represents the likelihood of there being sequential crimes in
        those two areas.

        This function assumes the list of crimes is already sorted by date.
    """
    # set up the graph with all of the vertices
    g = igraph.Graph()
    for area in areas:
        g.add_vertex(**area)

    last_vertex = None
    count = 0
    for c in crime_list:
        # simplify the graph after every thousand to make adding more edges faster
        if count % 1000 is 0:
            g.simplify(combine_edges=sum)
            print(count)
        count += 1
        # link to the previous vertex
        if last_vertex is not None:
            # find what area it is in
            area = get_index_of_crime(areas, crime_in_area, c)
            g.add_edge(last_vertex, area, weight=1)
            last_vertex = area
        else:
            last_vertex = get_index_of_crime(areas, crime_in_area, c)

    g.simplify(combine_edges=sum)
    return g


def get_index_of_crime(areas, crime_in_area, crime):
    """ Gets the index of the area the given crime is in.
    """
    # search through all the nodes in the graph
    for index in range(len(areas)):
        if crime_in_area(crime, areas[index]):
            return index
    return None