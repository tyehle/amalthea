import pymongo
import igraph
import math

# set up the connection to the db
client = pymongo.MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes  # get the reference to the collection
zipcodes = db.zipcodes


# TODO: Change this crap to generate nodes based on a data set
# TODO: Docstrings


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
        zs = [data for data in zipcodes.find({'city': name, 'state': state})]

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


def sequential_crime_graph(crime_window):
    """ Returns a graph where each node represents a crime, and each sequential
        crime by date is connected by an edge.
    """

    # sort the given list of crimes by date
    crime_window = sorted(crime_window, key=lambda x: x['date'])

    # make a graph
    g = igraph.Graph()

    # iterate through the graph, linking each node to the previous one
    last_vertex = None
    for c in crime_window:
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
        zips = [data for data in zipcodes.find({'city': city, 'state': state})]

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


def time_distance_crime_graph(crime_window, dist, time):
    """ Returns a graph object of crime occurrences.

        Creates graph of given crimes where edges arise when crimes occurred within
        dist miles of each other and within the specified time of each other.

        Parameters
        ----------
        crime_window: list
           List of crimes retrieved from crimes database.
        dist: float
            Distance of interest in miles.
        time: datetime.timedelta
            Time of interest in preferred units.

        Returns
        -------
        igraph.Graph
            Graph with vertices indicating crimes and edges indicating crimes
            that occurred within the given time window and distance window.

        Examples
        --------
        >>> import windows
        >>> from datetime import timedelta
        >>> c = windows.crime_window(cities = ['Greensboro'], states = ['NC'], crime_types = ['Assault'])
        >>> g = time_distance_graph(c, 2, timedelta(minutes = 15))
        >>> print(g.summary())
        IGRAPH U--T 1672 104 -- + attr: date (v), latitude (v), longitude (v),
        type (v), zipcode (v)
    """
    g = igraph.Graph()
    # Traverse crimes in window, for each crime create a vertex
    for crime in crime_window:
        g.add_vertices(1)
        c = g.vcount() - 1
        g.vs[c]["date"] = crime["date"]   # TODO: Rewrite using **
        g.vs[c]["zipcode"] = crime["zipcode"]
        g.vs[c]["latitude"] = float(crime["latitude"])
        g.vs[c]["longitude"] = float(crime["longitude"])
        g.vs[c]["type"] = crime["type"]
        # Connect vertices within dist miles and time minutes of each other
        for v in range(c):
            # Calculate distance between vertex v and c
            d_delta = v_distance(g.vs[c]["latitude"], g.vs[c]["longitude"],
                                 g.vs[v]["latitude"], g.vs[v]["longitude"])
            t_delta = abs(g.vs[c]["date"] - g.vs[v]["date"])
            if d_delta <= dist and t_delta <= time:
                g.add_edge(c, v)
    return g


def v_distance(lat1, lon1, lat2, lon2):
    """ Return distance between the two given locations in miles.

        Parameters
        ----------
        lat1, lon1, lat2, lon2: float
            Latitude/longitude location for point 1 and point 2.

        Returns
        -------
        float
            Distance between point 1 and point 2 in miles.

        References
        ----------
        Formula from: http://stackoverflow.com/questions/4913349/haversine-
        formula-in-python-bearing-and-distance-between-two-gps-points

        Examples
        --------
        >>> v_distance(36.876, -80.911, 32.543, -88.890)
        543.1390838847607
    """

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 3961  # Earth's radius in miles
    return abs(r * c)


def distance_zip_graph(crime_list, distance):
    for c in crime_list:
        c['latitude'] = float(c['latitude'])
        c['longitude'] = float(c['longitude'])
    first = lambda cs: cs[0]
    mean = lambda cs: sum(cs)/float(len(cs))
    return get_graph(crime_list,
                     lambda a, b: v_distance(a['latitude'],
                                             a['longitude'],
                                             b['latitude'],
                                             b['longitude']) < distance,
                     lambda c: c['zipcode'],
                     {'zipcode': first,
                      'latitude': mean,
                      'longitude': mean})


def distance_crime_graph(crime_list, distance):
    for c in crime_list:
        c['latitude'] = float(c['latitude'])
        c['longitude'] = float(c['longitude'])
    first = lambda cs: cs[0]
    return get_graph(crime_list,
                     lambda a, b: v_distance(a['latitude'],
                                             a['longitude'],
                                             b['latitude'],
                                             b['longitude']) < distance,
                     lambda c: c['_id'],
                     {'description': first, 'zipcode': first,
                      'longitude': first, 'latitude': first,
                      'address': first, 'date': first, 'type': first})


def get_graph(crime_list, crime_associated, get_id, combination_rules, add_index=False):
    """ Builds a graph from a list of crimes with edges and nodes based on the
        given rules.

        The generated graph uses the data from `crime_list` to generate nodes.
        An edge between two nodes has a weight equal to the number of
        associated crimes in the two nodes.

        Parameters
        ----------
        crime_list : list of dict
            The list of crimes used to build the graph. Each crime is
            represented as a dictionary of attributes and their values.
        crime_associated : (dict, dict -> bool)
            This function takes two crimes represented as dictionaries and
            determines if they should be associated. In addition to the
            attributes of the crimes, the dictionaries passed to this function
            also contain the index the crimes appeared in the crime list.
            ie. The first crime in the list would be passed as
            {'type': 'Theft', 'index': 0}
        get_id : (dict -> id)
            This function should take a crime as a dictionary and return a
            comparable id object. Any two crimes that have the same id object
            will be represented in a single vertex.
        combination_rules : dict<string,(list<val> -> val)>
            This dictionary defines the rules used to collapse many vertices
            with the same id into a single vertex. Keys should be the same as
            the keys in the crime_list, and the values should be functions
            taking a list of values from the crime list and returning a single
            value for use in the final graph.
        add_index : boolean
            Specifies if the index is needed in the association function.

        Returns
        -------
        graph : igraph.Graph
            A graph constructed using the data in `crime_list` and the rules
            defined by the other parameters.

        Notes
        -----
        The complexity of the algorithm is O(N^2 * k) where N is the number of
        crimes in the crime_list, and k is the complexity of
        `crime_associated`.

        Examples
        --------
        >>> from windows import crime_window
        >>> cs = crime_window(max_size=30)
        >>> cs = sorted(cs, key=lambda c: c['date'])
        >>> first = lambda x: x[0]
        >>> seq_g = get_graph(cs,
        ...                   lambda a, b: abs(a['index'] - b['index']) is 1,
        ...                   lambda c: c['type'],
        ...                   {'type': first},
        ...                   add_index=True)
        >>> type(seq_g)
        <class 'igraph.Graph'>
        >>> mean = lambda x: sum(x)/float(len(x))
        >>> dist_g = get_graph(cs,
        ...                    lambda a, b: v_distance(float(a['latitude']),
        ...                                            float(a['longitude']),
        ...                                            float(b['latitude']),
        ...                                            float(b['longitude'])) < 100,
        ...                    lambda c : c['type'],
        ...                    {'type': first,
        ...                     'latitude': mean,
        ...                     'longitude': mean})
        >>> type(dist_g)
        <class 'igraph.Graph'>
    """
    if add_index:
        crime_list = [dict(crime_list[i], index=i) for i in range(len(crime_list))]

    # generate the mapping from crime
    new_ids = [get_id(c) for c in crime_list]
    unique_ids = list(set(new_ids))
    id_map = {unique_ids[i]: i for i in range(len(unique_ids))}
    indices = [id_map[new_id] for new_id in new_ids]

    # create a mapping of edges as tuples to their weights
    edges = dict()
    for i in range(len(crime_list)):
        if i % 100 is 0:
            print('{0} / {1} : {2} edges'.format(i, len(crime_list), len(edges)))

        # add edges to the current vertex
        for p in range(i+1, len(crime_list)):
            # add edges for everything that should be associated
            if crime_associated(crime_list[p], crime_list[i]):
                edge = (indices[p], indices[i])
                if edge in edges:
                    edges[edge] += 1
                else:
                    edges[edge] = 1

    node_attributes = [dict() for _ in unique_ids]
    # create the list of node attributes
    for attribute, function in combination_rules.iteritems():
        for i in range(len(unique_ids)):
            crimes_with_id = [crime_list[j] for j in range(len(crime_list)) if indices[j] == i]
            node_attributes[i][attribute] = function([c[attribute] for c in crimes_with_id])

    g = igraph.Graph()
    for attrs in node_attributes:
        g.add_vertex(**attrs)

    g.add_edges(edges.iterkeys())
    g.es['weight'] = [float(w) for w in edges.itervalues()]

    return g