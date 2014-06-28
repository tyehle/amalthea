__author__ = 'tobin'

import logging.config
import math
import multithreading
import pymongo
import network_creation
import windows
from random import random
import json
from shapely.geometry import shape
from shapely.ops import cascaded_union

_earth_rad = 3956.5467  # miles
_multiprocess = False

logger = logging.getLogger(__name__)


def get_client():
    """ Gets a new instance of the pymongo client object. """
    return pymongo.MongoClient('163.118.78.22', 27017)


def get_bounds(city):
    """ Finds a set of bounding coordinates using the shapes of the zip codes
        for a city.

        :param city: The name of a city in the cities.json file
        :return: The bounds as a tuple of (left, bottom, right, top)

        Examples
        --------
        >>> left, bottom, right, top = get_bounds('miami')
    """
    zips = json.load(open('cities.json', 'r'))[city]
    geoms = get_client().crimes.geometry
    shapes = [shape(s['geometry']) for s in geoms.find({'zip': {'$in': zips}})]
    return cascaded_union(shapes).bounds


def distance_crime_network_by_box(distance, city, box_size=20, limits=None, crime_limit=100000.0):
    """ Builds a network for a city by merging many smaller networks.

        The module level flag, `_multiprocess` determines if the construction
        of the network will happen in different process or in serial on a
        single process.

        :param distance: The maximum distance between two connected crimes.
        :param city: The name of the city to find crimes in. There should be an
        entry in cities.json for the given name.
        :param box_size: The approximate length of a side of the sub-networks
        in miles. This will be rounded so there is an integer number of boxes
        of the same size.
        :param limits: Any additional limits to be passed to the crime_window
        function. These may include dates or crime types, eg: {'type': 'Theft'}
        :param crime_limit: An approximate limit on the number of crimes in the
        network. Very large networks may make some systems run out of memory.
        If that happens this value can be lowered. This limit is used to define
        a chance of any crime being included in the network, thus the exact
        number of crimes in the final network is not known until the network is
        complete.
        :return: A network of crimes for the given area and distance with
        approximately the given number of nodes.

        Examples
        --------
        >>> g = distance_crime_network_by_box(1.6, 'miami', limits={'type': 'Assault'})
    """
    left, bottom, right, top = get_bounds(city)
    logger.info('distance={}, lon=({} to {}), lat=({} to {})'.format(distance,
                                                                     left,
                                                                     right,
                                                                     bottom,
                                                                     top))

    cs = get_client().crimes.crimes
    bounds = {'$within': {'$box': [[left, bottom], [right, top]]}}
    if limits is not None:
        count = cs.find(dict(limits, coordinates=bounds)).count()
    else:
        count = cs.find({'coordinates': bounds}).count()
    keep_frac = crime_limit / count
    if keep_frac > 1:
        keep_frac = 1
    logger.info('{} crimes in area: keep chance {}'.format(count, keep_frac))

    conv = math.pi/180.0

    row_overlap = distance/_earth_rad / conv

    bottom_rad = math.cos(bottom * conv) * _earth_rad
    top_rad = math.cos(top * conv) * _earth_rad

    logger.debug('top radius: {}, bottom radius: {}'.format(top_rad, bottom_rad))

    width = right - left
    height = top - bottom

    logger.debug('width: {}, height: {}'.format(width, height))

    num_box_x = int(math.ceil(width*conv*bottom_rad / box_size))
    num_box_y = int(math.ceil(height*conv*_earth_rad / box_size))

    logger.info('{} boxes wide, {} boxes tall'.format(num_box_x, num_box_y))

    boxes = get_box_networks(distance, bottom, left, width, height,
                             num_box_x, num_box_y, limits, keep_frac)

    logger.debug('{} total edges'.format(reduce(lambda s, n: s+n.ecount(), boxes.values(), 0)))

    rows = stitch_boxes(boxes, distance)

    logger.info('{} total crimes'.format(reduce(lambda s, n: s+n.vcount(), rows, 0)))
    logger.info('{} total edges'.format(reduce(lambda s, n: s+n.ecount(), rows, 0)))

    return stitch_rows(rows, distance, row_overlap)


def stitch_boxes(networks, distance):
    """ Turns box networks into a list of rows.

        :param networks: A map<(x, y), network>, where (x, y) is the index of
        the box.
        :param distance: The maximum distance between linked crimes.
        :return: A list of row networks. Each row is composed of all boxes with
        the same y index. The list of rows is ordered from bottom to top.
        ie: the index of a row in the list is the same as its y index in the
        input map.
    """
    max_x = max([k[0] for k in networks.keys()])
    max_y = max([k[1] for k in networks.keys()])
    params = []
    for y in range(max_y + 1):
        params.append({'network_row': [networks[(x, y)] for x in range(max_x + 1)],
                       'distance': distance,
                       'row_number': y})

    # stitch into rows
    logger.info('Stitching into {} rows'.format(len(params)))
    if _multiprocess:
        rows = multithreading.map_kwargs(make_row_network, params, failsafe=True)
    else:
        rows = map(lambda args: make_row_network(**args), params)
    # make sure no rows died
    if False in rows:
        logger.fatal('Some Rows Failed!')
        raise RuntimeError
    return rows


def get_box_networks(distance, bottom, left, width, height,
                     num_box_x, num_box_y, limits=None, keep_frac=1):
    """ Gets a number of networks contained in boxes.

        :param distance: The maximum distance between connected nodes
        :param bottom: The bottom coordinate of the group of boxes
        :param left: The left coordinate of the group of boxes
        :param width: The total width of the group of boxes
        :param height: The total height of the group of boxes
        :param num_box_x: The number of boxes in a row
        :param num_box_y: The number of boxes in a column
        :param limits: Any restrictions on database output
        eg: {'type': 'Theft'}
        :return: A map from the (x,y) index of a box to the network of crimes
        in the box.
    """
    box_width = width / num_box_x
    box_height = height / num_box_y

    params = multithreading.combinations(width=[box_width],
                                         height=[box_height],
                                         x=range(num_box_x),
                                         y=range(num_box_y),
                                         gl_bottom=[bottom],
                                         gl_left=[left],
                                         limits=[limits],
                                         distance=[distance],
                                         keep_frac=[keep_frac])

    logger.debug('Broke into {} boxes'.format(len(params)))

    if _multiprocess:
        results = multithreading.map_kwargs(get_box_network, params, failsafe=True)
    else:
        results = map(lambda args: get_box_network(**args), params)

    # fail if we are missing boxes
    if False in results:
        logger.fatal('Some Boxes Failed!')
        raise RuntimeError

    # get a map<(x,y), network>
    # I know dict is ghetto, 2d array would be better
    return {(params[i]['x'], params[i]['y']): results[i] for i in range(len(results))}


def stitch_rows(rows, distance, row_overlap):
    """ Stitches a number of rows into a single network.

        It is assumed the rows are ordered bottom to top in the given list.
        This method does log(N) parallel row merges to stitch the network.

        :param rows: The list of rows to stitch
        :param distance: The distance to connect edges
        :param row_overlap: The maximum distance into another network needed to
        check for edges. This value is in degrees.
        :return: A single unified network
    """
    finished = False
    while not finished:
        logger.debug('{} crimes in {} rows'.format(reduce(lambda s, n: s+n.vcount(), rows, 0),
                                                   len(rows)))
        # stitch every other row together
        params = []
        i = 0
        while i+1 < len(rows):
            base = rows[i]
            i += 1
            top = rows.pop(i)
            params.append({'base': base,
                           'other': top,
                           'distance': distance,
                           'overlap': row_overlap})

        logger.info('Stitching into {} rows ({} stitches)'.format(len(rows), len(params)))

        if _multiprocess:
            results = multithreading.map_kwargs(stitch_two_rows, params, failsafe=True)
        else:
            results = map(lambda args: stitch_two_rows(**args), params)

        if i == len(rows) - 1:
            results.append(rows[i])
        rows = results

        # check we didn't throw up on a row
        if False in results:
            logger.fatal('Some Rows Failed!')
            raise RuntimeError

        finished = len(rows) == 1

    return rows[0]


def stitch_two_rows(base, other, distance, overlap):
    """ Called in parallel to stitch rows together and return the result. """
    stitch_networks(base, other, distance, overlap, 'latitude', 'top')
    return base


def get_box_network(width, height, x, y, gl_bottom, gl_left, limits, distance, keep_frac=1):
    """ Constructs a single network of all crimes in a box.

        :param width: The width of a box in degrees.
        :param height: The height of a box in degrees.
        :param x: The x index of the box to construct.
        :param y: The y index of the box to construct.
        :param gl_bottom: The latitude of the bottom of the first box.
        :param gl_left: The longitude of the right side of the first box.
        :param limits: Any additional limits on the crimes used to construct
        the network, eg. {'type': 'Theft'}
        :param distance: The maximum distance between two associated crimes.
        :return: A network composed of all crimes in the box and provided
        limits. The network has the top, bottom, left, and right attributes
        set to the bounding coordinates of the box.
    """
    if _multiprocess:
        # this will be run in parallel, so change the name of the logger
        logger.name = '< {} : {} >'.format(x, y)
    else:
        logger.debug('< {} : {} >'.format(x, y))

    # find the bounds for this box
    bottom = gl_bottom + height*y
    top = gl_bottom + height*(y+1)
    left = gl_left + width*x
    right = gl_left + width*(x+1)

    crimes = get_client().crimes.crimes
    if limits is None:
        limits = {'coordinates': {'$within': {'$box': [[left, bottom], [right, top]]}}}
    else:
        limits = dict(limits, coordinates={'$within': {'$box': [[left, bottom], [right, top]]}})
    logger.debug('Querying {}'.format(limits))
    data = windows.normalize_data(crimes.find(limits))
    data = [d for d in data if random() < keep_frac]  # remove ~ 1-keep_frac from the data
    logger.debug('{} crimes found'.format(len(data)))
    g = network_creation.distance_crime_graph(data, distance)
    # save the dimensions of this network
    g['top'] = top
    g['bottom'] = bottom
    g['left'] = left
    g['right'] = right
    return g


def make_row_network(network_row, distance, row_number):
    """ Stitches a number of box networks into a row network.

        :param network_row: A list of box networks ordered from left to right.
        :param distance: The maximum distance between two linked nodes.
        :param row_number: The row index. This is used to rename the logger for
        more comprehensible output if run in parallel.
        :return: A single network representing the whole row of box networks
    """
    total = len(network_row)
    row = network_row.pop(0)
    if _multiprocess:
        logger.name = '< {} >'.format(row_number)

    # find the maximum required overlap
    conv = math.pi/180.0
    small_rad = math.cos(max(abs(row['top']), abs(row['bottom'])) * conv) * _earth_rad
    overlap = distance/small_rad / conv

    while len(network_row) > 0:
        right = network_row.pop(0)
        stitch_networks(row, right, distance, overlap, 'longitude', 'right')
        logger.debug('{} / {}'.format(total - len(network_row), total))

    return row


def stitch_networks(base, other, distance, overlap, overlap_axis, base_edge):
    """ Stitches two networks together.

        :param base: The base network. All nodes and edges from the other
        network will be added to this one, and the nodes will be stitched with
        any additional edges that may be required.
        :param other: The network to add to the base network. This method does
        not cause any side effects in this network.
        :param distance: The maximum distance between two connected nodes.
        :param overlap: The length of overlap between the two networks in which
        nodes could be connected.
        :param overlap_axis: The axis on which the stitch is taking place.
        Should be one of either 'latitude' or 'longitude'.
        :param base_edge: The edge
        :return:
    """
    logger.debug('Stitching {} at {}'.format(base_edge, base[base_edge]))
    stitch_edges = []
    for bi in range(base.vcount()):
        base_v = base.vs[bi]
        # only check the vertex if it is in the overlap
        if base_v[overlap_axis] >= base[base_edge] - overlap:
            for i in range(other.vcount()):
                if other.vs[i][overlap_axis] <= base[base_edge] + overlap:
                    if network_creation.within_distance(base_v, other.vs[i], distance):
                        stitch_edges.append((bi, i+base.vcount()))
    # copy vertices from other into base
    offset = base.vcount()
    for v in other.vs:
        base.add_vertex(**v.attributes())
    # copy the edges
    # NOTE: Assumes the edges have no properties. eg: weight
    es = [(e[0]+offset, e[1]+offset) for e in other.get_edgelist()]
    base.add_edges(es)

    # add the stitch edges
    base.add_edges(stitch_edges)

    logger.debug('Done stitching {} at {}'.format(base_edge, base[base_edge]))
    # update the bounds of base
    base[base_edge] = other[base_edge]


if __name__ == "__main__":
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    # logging.basicConfig(level=logging.DEBUG)

    _distance = 0.8
    _city = 'los_angeles'

    import time
    from datetime import datetime
    start = time.time()
    _limits = {'date': {'$gte': datetime(2010, 12, 24)}}
    # _limits = None
    _network = distance_crime_network_by_box(_distance, _city, box_size=8, limits=_limits)
    logger.info('{} seconds'.format(time.time() - start))
