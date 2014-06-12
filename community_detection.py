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
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import multiprocessing
import os.path
import json
import itertools
import logging.config

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
    return igraph.Graph.Read('{}/networks/{}.graphml'.format(path, filename))


def _add_regions(g, cells_file, create_and_add):
    """ Adds regions to each node in a network.

        The regions are stored as shapely.Polygon objects as the cell attribute
        in each vertex of the network. If no regions can be found on disk the
        create_and_add function is used to make new ones. These are then saved
        to disk for future use.

        :param g: The network to add regions to.
        :param cells_file: The file regions should be stored in.
        :param create_and_add: A function to create and add regions if there
        are none on file.

        Examples
        --------
        >>> path = 'data/testing'
        >>> name = 'test'
        >>> g = load_network(path, name)
        >>> cells_file = '{}/regions/zip/{}.shp'.format(path, name)
        >>> def new_zip_regions(gr):
        ...     geometry = _client['crimes'].geometry
        ...     gr.vs['cell'] = [geometry.find_one({'zip': node['zipcode']}) for node in gr.vs]
        >>> _add_regions(g, cells_file, new_zip_regions)
        >>> 'cell' in g.vs.attributes()
        True
    """
    if os.path.exists(cells_file):
        logger.info("Loading Regions")
        # load cells from file
        for p in fiona.open(cells_file):
            g.vs[p['properties']['index']]['cell'] = shapely.geometry.shape(p['geometry'])
    else:
        logger.info("No Regions Found")

        create_and_add(g)

        logger.info("Saving Regions")

        # save cells for future use
        ensure_folder(cells_file)
        schema = {'geometry': 'Polygon',
                  'properties': {'index': 'int'}}
        with fiona.open(cells_file, 'w', 'ESRI Shapefile', schema) as c:
            for i in range(g.vcount()):
                writable = shapely.geometry.mapping(shapely.geometry.shape(g.vs[i]['cell']))
                c.write({'geometry': writable,
                         'properties': {'index': i}})


def add_zip_regions(g, path, filename, parallel=False):
    """ Adds zipcode based regions to a network.

        Uses the path and filename arguments to find any regions that may be
        stored on disk. If none are found the ones created for this network
        will be stored for future use.

        :param g: The network to add regions to.
        :param path: The base path to the network.
        :param filename: The filname of the network.

        Examples
        --------
        >>> path = 'data/testing'
        >>> file = 'test'
        >>> g = load_network(path, file)
        >>> add_zip_regions(g, path, file)
        >>> 'cell' in g.vs.attributes()
        True
    """
    cells_file = '{}/regions/zip/{}.shp'.format(path, filename)
    def new_zip_regions(gr):
        geometry = _client['crimes'].geometry
        gr.vs['cell'] = [geometry.find_one({'zip': node['zipcode']}) for node in gr.vs]

    _add_regions(g, cells_file, new_zip_regions)


def add_voronoi_regions(g, path, filename, parallel=True):
    """ Adds Voronoi cell based regions to a network.

        Uses the path and filename arguments to find any regions that may be
        stored on disk. If none are found the ones created for this network
        will be stored for future use.

        :param g: The network to add regions to.
        :param path: The base path to the network.
        :param filename: The filname of the network.

        Examples
        --------
        >>> path = 'data/testing'
        >>> file = 'test'
        >>> g = load_network(path, file)
        >>> add_voronoi_regions(g, path, file)
        >>> 'cell' in g.vs.attributes()
        True
    """
    cells_file = '{}/regions/voronoi/{}.shp'.format(path, filename)

    def new_voronoi(gr):
        layout_position(gr)
        bound = get_bounds(gr)
        logger.info('Creating Voronoi Cells')
        create_voronoi_regions(gr, bound)
        logger.info('Clipping Cells')
        clip_cells(gr, bound, parallel=parallel)

    _add_regions(g, cells_file, new_voronoi)


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

        if x is not -1 and y is not -1:
            # this is a bounded ridge
            continue

        # determine which point is unbounded
        unbounded = 0 if x is -1 else 1
        bounded = 1 if x is -1 else 0

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
            if inf_i is 0 and close_i is len(r)-1 or \
               inf_i is len(r)-1 and close_i is 0:
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


class _Clipper(object):
    """ A pickleable function object used to clip polygons.

        The multiprocessing module pickles the function it calls, so we need a
        class definition instead of just a function.
    """
    def __init__(self, bounds):
        """ Makes a clipping object that will clip polygons to the bounds given
            here.
            :param bounds: The bounds to clip polygons to
        """
        self.bounds = bounds

    def __call__(self, polygon):
        """ Clips the given polygon.
            :param polygon: The polygon to clip to the predefined bounds
            :return: The intersection of the bounds and the given polygon
        """
        return self.bounds.intersection(polygon)


def clip_cells(g, bounds, parallel=True):
    """ Clips all the cells in a graph to a bounding polygon.

        This algorithm is run in parallel because it may take a while if there
        are many vertices in the given graph.

        :param g: The graph containing nodes with cells to clip. Each vertex
        should have a 'cell' attribute representing the area of influence of
        the cell.

        :param bounds: The bounds to clip the cells to. This should be a
        polygon covering all the area cells could potentially be in.

        :param parallel: Whether to do this in parallel or not

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> create_voronoi_regions(g, bounds)
        >>> clip_cells(g, bounds)
    """
    if parallel:
        # run this in parallel
        pool = multiprocessing.Pool()
        g.vs['cell'] = pool.map(_Clipper(bounds), g.vs['cell'])
    else:
        g.vs['cell'] = map(_Clipper(bounds), g.vs['cell'])


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
    if os.path.exists(cluster_path):
        cluster_sets = json.load(open(cluster_path, 'r'))
    else:
        cluster_sets = []
    logger.info('Loaded {} communities'.format(len(cluster_sets)))
    # add new clusters if needed
    while len(cluster_sets) < n:
        logger.debug('{} / {} communities'.format(len(cluster_sets), n))
        clustering = _algorithms[algorithm](g)
        cluster_sets.append({'membership': clustering.membership,
                             'modularity_params':clustering._modularity_params})
    # save the cluster sets
    ensure_folder(cluster_path)
    json.dump(cluster_sets, open(cluster_path, 'w'))

    # construct a list of objects
    clusters = [igraph.VertexClustering(g, **c) for c in cluster_sets]
    return clusters[:n]  # return only the first n


def _collapse_duplicate_areas(temp_list):
    """ Collapses any duplicate areas in the given dictionary into a single
        key with a greater weight.

        Takes list<shapes>
        Returns map<shape, int>
    """
    # Reduce all borders to unique Polygons
    line_dict = dict()
    while len(temp_list) > 0:
        p = temp_list.pop()
        # convert polygon to line
        if p.geom_type == 'Polygon':
            p = [shapely.geometry.LineString(list(p.exterior.coords))]
        elif p.geom_type == 'MultiPolygon':
            p = [shapely.geometry.LineString(list(pi.exterior.coords)) for pi in p]
        else:
            logger.warn('Unrecognized region geometry {}, ignoring'.format(p.geom_type))

        for poly in p:
            exists = False
            for q in line_dict.keys():
                if poly.equals(q):
                    line_dict[q] += 1
                    exists = True
            if not exists:
                line_dict[poly] = 1

    return line_dict


def _separate_borders(line_dict):
    """ Separates the borders defined by the given dictionary into individual,
        non-intersecting line segments. The value in the dictionary represents
        the number of times the given segment appeared in the original list.

        Takes map<shape, int>
        Returns map<shape, int>
    """
    # Assign weights
    unique_borders = dict()
    done = False
    while not done:
        done = True
        logger.debug('{} / {} borders'.format(len(unique_borders), len(line_dict)))
        # iterate through the current version of line_dict
        try:
            for border in line_dict:
                if not done:
                    # if we have changed the dictionary do not try to continue
                    break
                # see if this border intersects any other borders in the dict
                for other_border in line_dict:
                    if not done:
                        # if we have changed the dictionary do not try to continue
                        break
                    if border is other_border:
                        # these are the same border, ignore
                        continue
                    # check for intersection
                    i = border.intersection(other_border)
                    if i.length > 0.0:
                        logger.debug('adjusting borders')
                        # corrupt the dictionary
                        diff_border = border.difference(i)
                        if diff_border.length > 0.0:
                            # if there is nothing here do not add it to the dict
                            line_dict[diff_border] = line_dict[border]
                        diff_other_border = other_border.difference(i)
                        if diff_other_border.length > 0.0:
                            line_dict[diff_other_border] = line_dict[other_border]

                        line_dict[i] = line_dict[border] + line_dict[other_border]
                        del line_dict[border]
                        del line_dict[other_border]
                        # its innocence is gone forever
                        done = False
                if done:
                    # this border does not intersect any other border
                    # remove it from line_dict so we don't check it anymore
                    unique_borders[border] = line_dict[border]
                    del line_dict[border]
                    done = False
        except RuntimeError:
            # the dict changed size, but we don't care
            pass

    # Account for any possible points in the current borders
    for border in unique_borders.keys():
        if border.geom_type == 'GeometryCollection':
            bad_list = []
            for line in border:
                if line.geom_type == 'Point':
                    bad_list.append(line)
            # remove the offending border from the dict
            weight = unique_borders[border]
            del unique_borders[border]
            # fix the border
            for point in bad_list:
                border = border.difference(point)
            # add the healthy border back into the dict
            unique_borders[border] = weight

    # make sure none of the borders have area
    # this can happen if communities overlap?
    for border in unique_borders.keys():
        if border.area > 0:
            logger.warn('Border has area {}'.format(border))

    return unique_borders


def save_borders(path, filename, region_type='voronoi', iterations=10, algorithm='label_propagation', parallel=True):
    """ Saves a shapefile containing the borders for a given network.

        The network is found using the path and filename arguments. If there
        are already borders of the same type no new borders will be made.

        :param path: The base path to the network of interest.
        :param filename: The filename of the network of interest.
        :param region_type: The type of region surrounding each node.
        :param iterations: The number of iterations of community detection to
        use for border detection. More iterations will show weak borders more
        clearly.
        :param algorithm: The community detection algorithm to use.

        Examples
        --------
        >>> import os.path
        >>> save_borders('data/testing',
        ...              'test',
        ...              region_type='voronoi',
        ...              iterations=10,
        ...              algorithm='random_walk')
        >>> os.path.exists('data/testing/borders/voronoi/random_walk/test_10.shp')
        True
    """
    borders_path = '{}/borders/{}/{}/{}_{}.shp'.format(path, region_type, algorithm, filename, iterations)
    if os.path.exists(borders_path):
        # these borders already exist, no need to compute them again
        logger.info("Borders exist, skipping")
        return

    add_regions = {'zip': add_zip_regions,
                   'voronoi': add_voronoi_regions}

    g = load_network(path, filename)
    add_regions[region_type](g, path, filename, parallel)
    clusters = get_communities(g, iterations, path, filename, algorithm)

    communities = get_community_shapes(g, clusters)
    # unpack list of lists into a single list
    communities = [p for polys in communities for p in polys]

    logger.info('Reducing Borders')

    borders = quantify_borders(communities)

    logger.info('Saving Borders')

    ensure_folder(borders_path)
    schema = {'geometry': 'MultiLineString',
              'properties': {'weight': 'int'}}
    with fiona.open(borders_path, 'w', 'ESRI Shapefile', schema) as c:
        for border, w in borders.iteritems():
            c.write({'geometry': shapely.geometry.mapping(shapely.geometry.shape(border)),
                     'properties': {'weight': w}})


def get_community_shapes(g, clusters):
    """ Converts a list of igraph.VertexClustering objects into a list of
        shapes representing each community.

        :param g: The graph containing nodes with a cell attribute.
        :param clusters: A list of igraph.VertexClustering objects representing
        the sets of clusters.
        :return: A list of shapes, one for each community.

        Examples
        --------
        >>> g = load_network('data/testing', 'test')
        >>> iterations = 5
        >>> clusters = [g.community_multilevel(weights='weight') for _ in range(iterations)]
        >>> shapes = get_community_shapes(g, clusters)
        >>> num_comms = reduce(lambda s, c: s + len(c), clusters, 0)
        >>> len(shapes) == num_comms
        True
    """
    # get a list of shapes for each run of the clustering algorithm
    communities = [[] for _ in range(len(clusters))]
    for iteration in range(len(clusters)):
        comms = clusters[iteration]
        polys = [[] for _ in comms]  # group each region with the others in its community
        for i in range(len(comms)):
            for node_index in comms[i]:
                p = g.vs[node_index]['cell']
                if p.geom_type is "MultiPolygon":
                    polys[i].extend(p)
                else:
                    polys[i].append(p)
        # merge each list of polygons into a community
        communities[iteration] = map(cascaded_union, polys)
    return communities


def quantify_borders(communities):
    """ Breaks a list of polygons into a list of lines with weights.

        :param communities: A list of all the communities found in a network.
        :return: A map containing line segments as keys, and the number of
        times that unique segment appears in the community polygons as values.
    """
    line_dict = _collapse_duplicate_areas(communities)

    return _separate_borders(line_dict)


def plot_border_map(border_path, pad=.05):
    """ Plots the borders in a shapefile.

        :param border_path: The path to the shape file containing the borders
        to plot.
        :param pad: The percentage of the width to pad the edge of the map.
    """
    logger.info('Plotting Borders')
    fig = plt.figure()
    ax = fig.add_subplot()
    union = cascaded_union([shapely.geometry.shape(p['geometry']) for p in
                            fiona.open(border_path+'.shp')])
    (llx, lly, urx, ury) = union.bounds
    # assume we are in the northern hemisphere, west of the meridian
    width = urx - llx
    height = ury - lly
    llx -= pad * width
    urx += pad * width
    lly -= pad * height
    ury += pad * height
    m = Basemap(resolution='h',
                projection='merc',
                llcrnrlon=llx,
                llcrnrlat=lly,
                urcrnrlon=urx,
                urcrnrlat=ury,
                ax=ax)
    m.readshapefile(border_path, 'comm_bounds', drawbounds=False)
    m.drawmapboundary(color='k', linewidth=1.0, fill_color='#006699')
    m.fillcontinents(color='.6', zorder=0)
    colormap = plt.get_cmap('gist_heat')
    max_weight = max([border['weight'] for border in m.comm_bounds_info])
    for border, shape in zip(m.comm_bounds_info, m.comm_bounds):
        xx, yy = zip(*shape)
        m.plot(xx, yy, linewidth=2, color=colormap(border['weight'] / float(max_weight)))

    m.drawcoastlines(linewidth=1)

    # TODO: Make the color bar actually work
    # m.colorbar()

    return fig


class Worker(object):
    def __call__(self, params):
        try:
            city = params['city']
            distance = params['distance']
            node_type = params['node_type']
            region_type = params['region']
            algorithm = params['algorithm']
            name = params['filename']
            iterations = params['iterations']

            unique_id = '{}-{}-{}-{}-{}-{}-{}'.format(city,
                                                      distance,
                                                      node_type,
                                                      region_type,
                                                      algorithm,
                                                      name,
                                                      iterations)

            # The thread pool copies __main__, so this should change the name
            # of the logger for only this thread
            logger.name = unique_id

            logger.info('Starting!')

            path = 'data/{}/distance/{}/{}'.format(city, distance, node_type)

            add = {'voronoi': add_voronoi_regions, 'zip': add_zip_regions}

            network = load_network(path, name)

            add[region_type](network, path, name, parallel=False)

            _ = get_communities(network, iterations, path, name, algorithm=algorithm)

            save_borders(path,
                         name,
                         iterations=iterations,
                         algorithm=algorithm,
                         region_type=region_type,
                         parallel=False)

            fig = plot_border_map('{}/borders/voronoi/{}/{}_{}'.format(path,
                                                                       algorithm,
                                                                       name,
                                                                       iterations))
            fig.savefig('{}.svg'.format(unique_id))
            # fig.show()

            logger.info('Done!')
            return True
        except Exception:
            logger.error('Failed!', exc_info=True)


if __name__ == '__main__':
    # write info and debug to different files
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    fiona.log.setLevel(logging.WARNING)  # I don't care about the fiona logs

    # cities = ['los_angeles', 'baltimore']
    # distances = [3.2, 2.4, 1.6, .8, .1]
    # types = ['crime', 'zip']
    # algorithms = ['random_walk', 'eigenvector', 'label_propagation', 'fast_greedy', 'multilevel']
    # regions = ['voronoi', 'zip']
    # iterations = [30]
    # filenames = ['dec2010', '17dec2010', '30dec2010']

    cities = ['los_angeles']#, 'baltimore']
    distances = [3.2, 2.4, 1.6, .8, .1]
    types = ['zip']
    regions = ['voronoi']
    algorithms = ['label_propagation']
    filenames = ['30dec2010']# , '17dec2010', '30dec2010']
    iterations = [30]

    params = []
    for c, d, t, a, r, i, n in itertools.product(cities,
                                                 distances,
                                                 types,
                                                 algorithms,
                                                 regions,
                                                 iterations,
                                                 filenames):
        params.append({'city': c,
                       'distance': d,
                       'node_type': t,
                       'algorithm': a,
                       'region': r,
                       'iterations': i,
                       'filename': n})

    pool = multiprocessing.Pool()
    results = pool.map(Worker(), params)
    # results = map(Worker(), params)  # serial execution
    logger.info(results)
