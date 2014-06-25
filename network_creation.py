import pymongo
import igraph
import math
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# set up the connection to the db
client = pymongo.MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes  # get the reference to the collection
zipcodes = db.zipcodes


def reduce_to_zip_graph(crime_graph):
    """ Builds a zip graph based on the relations in the given crime graph.

        Warning: Any graph given to this method may change. Save a copy first!

        :param crime_graph: The crime graph to use as a seed for the zipcode
        graph. This graph's nodes must have a zipcode attribute and the edges
        must have a weight attribute.
        :return: The contracted graph. The weight attribute of the edges
        contains the sum of the weights from the original. Each node retains a
        zipcode, latitude, longitude, description, and type attribute. The
        description attribute is the number of contracted nodes in that vertex.
    """
    # generate the mapping from crime
    new_ids = crime_graph.vs['zipcode']
    unique_ids = list(set(new_ids))
    id_map = {unique_ids[i]: i for i in range(len(unique_ids))}
    index_map = [id_map[new_id] for new_id in new_ids]
    mode = lambda cs: Counter(cs).most_common(1)[0][0]
    crime_graph.contract_vertices(index_map,
                                  {'zipcode': 'first',
                                   'latitude': 'mean',
                                   'longitude': 'mean',
                                   'description': len,
                                   'type': mode})
    crime_graph.simplify(combine_edges=sum)


def distance_graph(crime_list, distance, node_type):
    if node_type == 'zip':
        return distance_zip_graph(crime_list, distance)
    elif node_type == 'crime':
        return distance_crime_graph(crime_list, distance)
    else:
        raise NotImplementedError('{} node networks not implemented'.format(node_type))


def distance_zip_graph(crime_list, distance):
    return get_graph(crime_list,
                     lambda a, b: within_distance(a, b, distance),
                     lambda c: c['zipcode'],
                     {'zipcode': 'first',
                      'latitude': 'mean',
                      'longitude': 'mean',
                      'description': len,
                      'type': 'mode'})


def distance_crime_graph(crime_list, distance):
    return get_graph(crime_list,
                     lambda a, b: within_distance(a, b, distance))


def within_distance(a, b, distance):
    earth_rad = 3956.5467  # miles
    lat1, lon1, lat2, lon2 = map(math.radians, [a['latitude'],
                                                a['longitude'],
                                                b['latitude'],
                                                b['longitude']])
    dlat = lat1 - lat2
    if dlat*earth_rad > distance:
        return False
    dlon = lon1 - lon2
    lon_scale = math.cos((lat1+lat2) / 2)
    if dlon*earth_rad*lon_scale > distance:
        return False
    d = earth_rad * math.sqrt((lon_scale * dlon)**2 + dlat**2)
    return d <= distance


def get_graph(attribute_list, is_associated, get_id=None, combination_rules=None, add_index=False):
    """ Builds a graph from a list of crimes with edges and nodes based on the
        given rules.

        The generated graph uses the data from `attribute_list` to generate nodes.
        An edge between two nodes has a weight equal to the number of
        associated crimes in the two nodes.

        Parameters
        ----------
        attribute_list : list of dict
            The list of crimes used to build the graph. Each crime is
            represented as a dictionary of attributes and their values.
        is_associated : (dict, dict -> bool or int)
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
            the keys in the attribute_list, and the values should be functions
            taking a list of values from the crime list and returning a single
            value for use in the final graph.
        add_index : boolean
            Specifies if the index is needed in the association function.

        Returns
        -------
        graph : igraph.Graph
            A graph constructed using the data in `attribute_list` and the rules
            defined by the other parameters.

        Notes
        -----
        The complexity of the algorithm is O(N^2 * k) where N is the number of
        elements in the attribute_list, and k is the complexity of
        `is_associated`.

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
        attribute_list = [dict(attribute_list[i], index=i) for i in range(len(attribute_list))]

    # generate the mapping from crime
    if get_id is None:
        unique_ids = indices = range(len(attribute_list))
    else:
        new_ids = [get_id(c) for c in attribute_list]
        unique_ids = list(set(new_ids))
        id_map = {unique_ids[i]: i for i in range(len(unique_ids))}
        indices = [id_map[new_id] for new_id in new_ids]

    # create a mapping of edges as tuples to their weights
    edges = dict()
    for i in range(len(attribute_list)):
        if i % 100 is 0:
            logger.debug('{0} / {1} : {2} edges'.format(i, len(attribute_list), len(edges)))

        # add edges to the current vertex
        for p in range(i+1, len(attribute_list)):
            # don't add self edges
            if indices[p] == indices[i]:
                continue
            # add edges for everything that should be associated
            weight = is_associated(attribute_list[p], attribute_list[i])
            if weight:
                edge = tuple(sorted([indices[p], indices[i]]))
                if edge in edges:
                    edges[edge] += weight
                else:
                    edges[edge] = weight

    common_functions = {'first': lambda cs: cs[0],
                        'last': lambda cs: cs[-1],
                        'mean': lambda cs: sum(cs)/float(len(cs)),
                        'mode': lambda cs: Counter(cs).most_common(1)[0][0]}
    if get_id is None:
        node_attributes = attribute_list
    else:
        node_attributes = [dict() for _ in unique_ids]
        # create the list of node attributes
        for attribute, function in combination_rules.iteritems():
            for i in range(len(unique_ids)):
                crimes_with_id = [attribute_list[j] for j in range(len(attribute_list)) if indices[j] == i]
                if function in common_functions:
                    node_attributes[i][attribute] = common_functions[function]([c[attribute] for c in crimes_with_id])
                else:
                    node_attributes[i][attribute] = function([c[attribute] for c in crimes_with_id])

    g = igraph.Graph()
    for attrs in node_attributes:
        g.add_vertex(**attrs)

    g.add_edges(edges.iterkeys())
    g.es['weight'] = [float(w) for w in edges.itervalues()]

    return g
