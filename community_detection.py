__author__ = 'Tobin Yehle'

import matplotlib
matplotlib.use('Agg')  # this fixes issues when executing over ssh
import igraph
import pymongo
from scipy.spatial import Voronoi
import numpy as np
from shapely.ops import cascaded_union
import shapely.geometry
import fiona
import os.path
import json
import multithreading
import logging.config
import network_creation
import plotting

_client = pymongo.MongoClient('163.118.78.22', 27017)
_algorithms = {'random_walk': lambda g: g.community_walktrap(weights='weight').as_clustering(),
               'eigenvector': lambda g: g.community_leading_eigenvector(),
               'label_propagation': lambda g: g.community_label_propagation(weights='weight'),
               'fast_greedy': lambda g: g.community_fastgreedy(weights='weight').as_clustering(),
               'multilevel': lambda g: g.community_multilevel(weights='weight')}
logger = logging.getLogger(__name__)


def ensure_folder(file_path):
    """ Ensures the folder structure exists for a file.

        :param file_path: The path to the file to ensure.

        Examples
        --------
        >>> import json
        >>> test = {'first': 10, 'second': 'foo'}
        >>> path = 'data/testing/foobar.json'
        >>> ensure_folder(path)
        >>> json.dump(test, open(path, 'w'))
    """
    d = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(d):
        os.makedirs(d)


def load_network(path, filename):
    """ Loads cached network for the filesystem.

        :param path: The base path to the network. This will contain all
        information about the type of the network.
        :param filename: The filename of the network. This is usually the dates
        of the data used to build the network.
        :return: an igraph.Graph representation of the network.

        Examples
        --------
        >>> path = 'data/lake_wobegon/distance/1.6/crime'
        >>> filename = '2010'
        >>> network = load_network(path, filename)
    """
    network_path = '{}/networks/{}.graphml'.format(path, filename)
    h = open(network_path, 'r')
    try:
        multithreading.lock_file_handle(h)
        return igraph.Graph.Read(network_path)
    finally:
        multithreading.unlock_file_handle(h)
        h.close()


def add_regions(g, path, filename, region_type):
    """ Adds regions to each node in a network.

        The regions are stored as shapely.Polygon objects as the cell attribute
        in each vertex of the network. If no regions can be found on disk the
        create_and_add function is used to make new ones. These are then saved
        to disk for future use.

        :param g: The network to add regions to.
        :param path: The path to the base folder of the network of interest.
        :param filename: The filename of the network of interest.
        :param region_type: The type of region to add to the graph. Should be
        one of: 'voronoi' or 'zip'.

        Examples
        --------
        >>> path = 'data/testing'
        >>> name = 'test'
        >>> g = load_network(path, name)
        >>> add_regions(g, path, name, 'zip')
        >>> 'cell' in g.vs.attributes()
        True
    """
    cells_file = os.path.join(path, 'regions', region_type, filename+'.shp')
    if os.path.exists(cells_file):
        logger.info("Loading Regions")
        h = open(cells_file[:-4]+'.dbf', 'r')
        try:
            multithreading.lock_file_handle(h)
            # load cells from file
            for p in fiona.open(cells_file):
                g.vs[p['properties']['index']]['cell'] = shapely.geometry.shape(p['geometry'])
        finally:
            multithreading.unlock_file_handle(h)
            h.close()
    else:
        logger.info("No Regions Found")

        # add the regions to the graph
        if region_type == 'voronoi':
            layout_position(g)
            bound = get_bounds(g)
            logger.info('Creating Voronoi Cells')
            create_voronoi_regions(g, bound)
            logger.info('Clipping Cells')
            clip_cells(g, bound)
        elif region_type == 'zip':
            logger.info('Finding Zipcode Cells')
            geometry = _client['crimes'].geometry
            g.vs['cell'] = [shapely.geometry.shape(geometry.find_one({'zip': node['zipcode']})['geometry']) for node in g.vs]
        else:
            logger.warning("Unrecognized region type: '{}'".format(region_type))
            raise NotImplementedError('{} regions not implemented'.format(region_type))

        logger.info("Saving Regions")

        # save cells for future use
        ensure_folder(cells_file)
        schema = {'geometry': 'Polygon',
                  'properties': {'index': 'int'}}
        h = open(cells_file[:-4]+'.dbf', 'a')
        try:
            multithreading.lock_file_handle(h)
            with fiona.open(cells_file, 'w', 'ESRI Shapefile', schema) as c:
                for i in range(g.vcount()):
                    writable = shapely.geometry.mapping(shapely.geometry.shape(g.vs[i]['cell']))
                    c.write({'geometry': writable,
                             'properties': {'index': i}})
        finally:
            multithreading.unlock_file_handle(h)
            h.close()


def layout_position(g):
    """ Sets the coordinates of each node in a graph using the latitude and
        longitude attributes.

        :param g: The graph to layout

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> layout_position(g)
        >>> 'x' and 'y' in g.vs.attributes()
        True
    """
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [float(y) for y in g.vs['latitude']]


def get_bounds(g):
    """ Gets a bounding polygon from the zipcodes in a graph.

        The bounds of the graph is the union of the areas of all the zipcodes
        in the graph.

        :param g: The graph to find the bounds of. Each node in the graph must
        have a zipcode attribute.

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> bounds.is_valid
        True
    """
    zips = list(set(g.vs['zipcode']))
    geometry = _client['crimes'].geometry
    results = []
    for z in zips:
        r = geometry.find_one({'zip': z})
        # we should have all zipcodes, but just in case ...
        if r is None:
            logger.warn('no associated zipcode shape for ' + z)
        else:
            results.append(r)

    shapes = [shapely.geometry.shape(r['geometry']) for r in results]
    return cascaded_union(shapes)


def _fix_unbounded_regions(vor, length):
    """ Makes all of the unbounded Voronoi cells bounded.

        For each unbounded cell this method creates an additional ridge
        connecting the two unbounded ridges at the given length from the finite
        vertex.
    """
    # find the center of the nodes
    center = vor.points.mean(axis=0)

    ### fix unbounded regions ###
    for i in range(len(vor.ridge_vertices)):
        (a, b) = vor.ridge_points[i]
        (x, y) = vor.ridge_vertices[i]

        if x != -1 and y != -1:
            # this is a bounded ridge
            continue

        # determine which point is unbounded
        unbounded = 0 if x == -1 else 1
        bounded = 1 if x == -1 else 0

        ### find the far point ###
        # find a vector tangent to the ridge
        t = vor.points[a] - vor.points[b]
        t /= np.linalg.norm(t)  # make t a unit vector
        # find a vector parallel to the ridge
        n = np.array([-t[1], t[0]])
        # correct the direction of the unit vector parallel to the ridge
        midpoint = (vor.points[a] + vor.points[b]) / 2.0
        n *= np.sign(np.dot(midpoint - center, n))
        # find the location of the far point
        far_point = vor.vertices[vor.ridge_vertices[i][bounded]] + n * length

        ### update the regions of a and b ###
        # update the correct ridge vertex
        vor.ridge_vertices[i][unbounded] = len(vor.vertices)
        # update the regions of points a and b
        for r in [vor.regions[vor.point_region[a]], vor.regions[vor.point_region[b]]]:
            close_i = r.index(vor.ridge_vertices[i][bounded])
            inf_i = r.index(-1)
            # case where close and inf are on the ends of the list
            if inf_i == 0 and close_i == len(r)-1 or \
               inf_i == len(r)-1 and close_i == 0:
                # put the new point at the end
                r.append(len(vor.vertices))
            else:
                # insert the new point between these two
                r.insert(max(close_i, inf_i), len(vor.vertices))
        # update the list of vertices
        vor.vertices = np.append(vor.vertices, [far_point], axis=0)

    # TODO: Add an additional point if the bounded region is too small
    # remove the -1 from any region that has it (can change to replace if needed)
    for r in vor.regions:
        try:
            r.remove(-1)
        except ValueError:
            # if this was a bounded region there won't be a -1
            pass


def create_voronoi_regions(g, bounds):
    """ Adds a 'cell' attribute to each node containing that node's Voronoi
        region.

        Uses 'x' and 'y' attributes of the nodes to calculate the positions of
        the Voronoi cells.

        :param g: The graph to add cells to
        :param bounds: The physical bounds of the graph. This is used to
        determine how large unbounded Voronoi cells should be.

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> create_voronoi_regions(g, bounds)
        >>> 'cell' in g.vs.attributes()
        True
    """
    # points = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    points = [(c['x'], c['y']) for c in g.vs]
    vor = Voronoi(points)

    # define the length of the unbounded ridges
    bounds_box = bounds.bounds
    length = bounds_box[2] - bounds_box[0] + bounds_box[3] - bounds_box[1]
    _fix_unbounded_regions(vor, length)

    # make polygons for each of the bounded regions
    # for i in vor.point_region:
    for point_i in range(len(vor.point_region)):
        region_i = int(vor.point_region[point_i])  # convert to regular ints to may igraph happy
        r = vor.regions[region_i]

        if -1 not in r:
            # this region is fully defined, so add it
            g.vs[point_i]['cell'] = shapely.geometry.Polygon([vor.vertices[j] for j in r])
        else:
            logger.warn('unbounded region {}'.format(r))


def clip_cells(g, bounds):
    """ Clips all the cells in a graph to a bounding polygon.

        :param g: The graph containing nodes with cells to clip. Each vertex
        should have a 'cell' attribute representing the area of influence of
        the cell.

        :param bounds: The bounds to clip the cells to. This should be a
        polygon covering all the area cells could potentially be in.

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> create_voronoi_regions(g, bounds)
        >>> clip_cells(g, bounds)
    """
    g.vs['cell'] = map(bounds.intersection, g.vs['cell'])


def get_communities(g, n, path, filename, algorithm='label_propagation'):
    """ Gets a number of igraph.VertexClustering objects.

        These objects are loaded from file if possible, otherwise they are
        found using the given algorithm.

        :param g: The graph to find communities in.
        :param n: The number of communities to find.
        :param path: The path to the base folder for the graph.
        :param filename: The filename of the graph to use.
        :param algorithm: The name of the clustering algorithm to use.

        The filename and path arguments are used to find clusters stored on
        disk. Any new clusters are stored along with the ones already present
        for future use.

        :return: A list of VertexClustering objects

        Examples
        --------
        >>> path = 'data/testing'
        >>> filename = 'test1'
        >>> g = load_network(path, filename)
        >>> comms = get_communities(g, 10, path, filename, algorithm='random_walk')
        >>> len(comms)
        10
    """
    # load any preexisting clusters
    cluster_path = '{}/communities/{}/{}.json'.format(path, algorithm, filename)
    ensure_folder(cluster_path)
    h = open(cluster_path, 'a')
    try:
        multithreading.lock_file_handle(h)
        try:
            cluster_sets = json.load(open(cluster_path, 'r'))
        except ValueError:
            # the file is probably empty because we just made it
            cluster_sets = []
        logger.info('Loaded {} communities'.format(len(cluster_sets)))
        # add new clusters if needed
        while len(cluster_sets) < n:
            logger.debug('{} / {} communities'.format(len(cluster_sets), n))
            clustering = _algorithms[algorithm](g)
            cluster_sets.append({'membership': clustering.membership,
                                 'modularity_params':clustering._modularity_params})
        # save the cluster sets
        json.dump(cluster_sets, open(cluster_path, 'w'))
    finally:
        multithreading.unlock_file_handle(h)
        h.close()

    # construct a list of objects
    clusters = [igraph.VertexClustering(g, **c) for c in cluster_sets]
    return clusters[:n]  # return only the first n


def get_adjacency_network(g, path, filename, region_type):
    """ Gets a network representing the physical adjacency of another network.

        :param g: The network to use as a base. The vertices of this network
        must have a cell attribute. If two cells have an intersection of
        non-zero length then they are considered adjacent.
        :param path: The base path to the network. The algorithm uses this path
        to cache temporary results.
        :param filename: The filename of the network. Also used for caching.
        :param region_type: The type of regions contained in the cell attribute
        of the base network. This is also used for caching.
        :return: An igraph.Graph object. All vertex attributes of the base
        network and the new network should be the same. Any attributes that
        cannot be written to a file by igraph (except cell) may not be present.

        Examples
        --------
        >>> path = 'data/testing'
        >>> filename = 'test'
        >>> region_type = 'zip'
        >>> g = load_network(path, filename)
        >>> add_regions(g, path, filename, region_type)
        >>> adj = get_adjacency_network(g, path, filename, region_type)
        >>> adj.vcount() == g.vcount()
        True
    """
    network_path = os.path.join(path, 'regions', region_type, filename+'.graphml')
    ensure_folder(network_path)
    if os.path.exists(network_path):
        logger.info('Loading Adjacency Network')
        h = open(network_path, 'r')
        try:
            multithreading.lock_file_handle(h)
            return igraph.Graph.Read(network_path)
        finally:
            multithreading.unlock_file_handle(h)
            h.close()
    else:
        logger.info('Creating Adjacency Network')
        info = [v.attributes() for v in g.vs]
        adj = network_creation.get_graph(info,
                                         lambda a, b: a['cell'].intersection(b['cell']).length > 0)
        h = open(network_path, 'a')
        try:
            multithreading.lock_file_handle(h)
            adj.write_graphml(network_path)
        finally:
            multithreading.unlock_file_handle(h)
            h.close()

        return adj


def find_border_weight(comms_runs, a, b):
    """ Finds the weight of the border between two nodes.

        Finds the number of times the two given nodes appear in different
        communities. This is the weight of the border between the two nodes,
        given they share a border.

        :param comms_runs: The list of clusters to search.
        :param a: The first node.
        :param b: The second node.
        :return: The weight of the border between the given nodes.
    """
    return reduce(lambda w, comms: w+1 if comms.membership[a] != comms.membership[b] else w,
                  comms_runs,
                  0)


def get_border_network(path, filename, region_type, algorithm, iterations):
    """ Finds a network representing the borders between communities.

        :param path: The base path to the network of crimes.
        :param filename: The filename of the network of crimes.
        :param region_type: The type of regions around each vertex.
        :param algorithm: The community detection algorithm to use.
        :param iterations: The number of runs of the community detection algorithm.
        :return: An `igraph.Graph` object where the weights of edges between
        two vertices represent the strength of a border between them.

        Examples
        --------
        >>> path = 'data/testing'
        >>> filename = 'test'
        >>> bn = get_border_network(path, filename, 'voronoi', 'label_propagation', 30)
        >>> bn.write_graphml('{}/borders/{}.graphml'.format(path, filename))
    """
    border_path = os.path.join(path, 'borders', region_type, algorithm,
                               '{}_{}.graphml'.format(filename, iterations))
    ensure_folder(border_path)
    if os.path.exists(border_path):
        logger.info('Loading Border Network')
        h = open(border_path, 'r')
        try:
            multithreading.lock_file_handle(h)
            border_network = igraph.Graph.Read(border_path)
        finally:
            multithreading.unlock_file_handle(h)
            h.close()
        add_regions(border_network, path, filename, region_type)
        return border_network
    else:
        # create a network showing the physical adjacency of each cell
        g = load_network(path, filename)

        add_regions(g, path, filename, region_type)

        border_network = get_adjacency_network(g, path, filename, region_type)
        add_regions(border_network, path, filename, region_type)

        if 'id' in border_network.vs.attributes():
            # get rid of it because it causes a warning
            del border_network.vs['id']

        # get the list of communities to use
        comms = get_communities(g, iterations, path, filename, algorithm)

        logger.info('Creating Border Network')
        # change the weights on the edges to reflect the border weight
        for e in border_network.es:
            e['weight'] = find_border_weight(comms, e.source, e.target)

        # save the network
        h = open(border_path, 'a')
        try:
            multithreading.lock_file_handle(h)
            border_network.write_graphml(border_path)
        finally:
            multithreading.unlock_file_handle(h)
            h.close()

        return border_network


def save_borders(path, filename, region_type, iterations, algorithm):
    """ Saves a shapefile containing the borders found a network.

        Finds a border network containing the information about any borders.
        Uses the cell attribute of the vertices and the weight attribute of the
        edges to build the shapes of the borders between communities.

        :param path: The base path to the network of crimes.
        :param filename: The filename of the crimes network.
        :param region_type: The type of region surrounding each vertex.
        :param iterations: The number of iterations of the community detection algorithm.
        :param algorithm: The community detection algorithm to use.

        Examples
        --------
        >>> import plotting
        >>> path = 'data/testing'
        >>> filename = 'test'
        >>> iterations = 30
        >>> save_borders(path, filename, 'zip', iterations, 'random_walk')
        >>> fig = plotting.get_border_fig('{}/borders/{}_{}'.format(path, filename, iterations))
        >>> fig.savefig('test.svg')
    """
    borders_path = os.path.join(path, 'borders', region_type, algorithm,
                                '{}_{}.shp'.format(filename, iterations))
    if os.path.exists(borders_path):
        # skip
        logger.info("Borders Exist, skipping")
        return
    border_network = get_border_network(path, filename, region_type, algorithm, iterations)
    borders = dict()
    for e in border_network.es:
        if e['weight'] > 0:
            line = border_network.vs[e.source]['cell'].intersection(border_network.vs[e.target]['cell'])

            # remove any points that might have snuck in
            if line.geom_type == 'GeometryCollection':
                points = [shp for shp in line if shp.geom_type == 'Point']
                for p in points:
                    line = line.difference(p)

            if line.geom_type == 'LineString' or line.geom_type == 'MultiLineString':
                borders[line] = e['weight']
            elif line.geom_type == 'Polygon' or line.geom_type == 'MultiPolygon':
                borders[line.boundary] = e['weight']
            else:
                logger.error('Unknown border geometry {}, skipping'.format(line.geom_type))

    ensure_folder(borders_path)
    schema = {'geometry': 'MultiLineString',
              'properties': {'weight': 'int'}}
    h = open(borders_path[:-4]+'.dbf', 'a')
    try:
        multithreading.lock_file_handle(h)
        with fiona.open(borders_path, 'w', 'ESRI Shapefile', schema) as c:
            for border, _w in borders.iteritems():
                c.write({'geometry': shapely.geometry.mapping(shapely.geometry.shape(border)),
                         'properties': {'weight': _w}})
    finally:
        multithreading.unlock_file_handle(h)
        h.close()


if __name__ == '__main__':
    # write info and debug to different files
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    fiona.log.setLevel(logging.WARNING)  # I don't care about the fiona logs

    params = multithreading.combinations(city=['los_angeles', 'baltimore'],
                                         distance=[3.2, 2.4, 1.6, .8, .1],
                                         node_type=['crime', 'zip'],
                                         region_type=['voronoi', 'zip'],
                                         algorithm=['random_walk', 'label_propagation'],
                                         filename=['dec2010', '17dec2010', '30dec2010'],
                                         iterations=[30])

    # filter out bad combinations
    params = filter(lambda d: d['node_type'] != 'crime' or d['region_type'] != 'zip', params)

    def work(city, distance, node_type, region_type, algorithm, filename, iterations):
        try:
            unique_id = '{}-{}-{}-{}-{}-{}-{}'.format(city,
                                                      distance,
                                                      node_type,
                                                      region_type,
                                                      algorithm,
                                                      filename,
                                                      iterations)

            # The thread pool copies __main__, so this will change the name
            # of the logger for only this thread
            logger.name = unique_id
            logger.info('Starting!')

            path = 'data/{}/distance/{}/{}'.format(city, distance, node_type)
            network = load_network(path, filename)
            add_regions(network, path, filename, region_type)

            _ = get_communities(network, iterations, path, filename, algorithm=algorithm)

            save_borders(path, filename, region_type, iterations, algorithm)

            figure_path = 'output/{}.svg'.format(unique_id)
            if not os.path.exists(figure_path):
                fig = plotting.get_border_fig('{}/borders/{}/{}/{}_{}'.format(path,
                                                                              region_type,
                                                                              algorithm,
                                                                              filename,
                                                                              iterations))
                fig.savefig(figure_path)
            else:
                logger.info('Figure Exists, skipping')

            logger.info('Done!')
            return True
        except:
            logger.error('Failed!', exc_info=True)
            return False

    _results = multithreading.map_kwargs(work, params)
    logger.info(_results)
    for _r in zip(params, _results):
        if not _r[1]:
            logger.info('{} => {}'.format(_r[0].values(), _r[1]))
