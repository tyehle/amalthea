__author__ = 'Tobin Yehle'

import igraph
import math
import pymongo
from scipy.spatial import Voronoi
import numpy as np
import shapely.ops
import shapely.geometry
import fiona
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import multiprocessing


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
    client = pymongo.MongoClient('163.118.78.22', 27017)
    geometry = client['crimes'].geometry
    results = []
    for z in zips:
        r = geometry.find_one({'zip': z})
        # we should have all zipcodes, but just in case ...
        if r is None:
            print('Warning: no associated zipcode shape for ' + z)
        else:
            results.append(r)

    shapes = [shapely.geometry.shape(r['geometry']) for r in results]
    return shapely.ops.cascaded_union(shapes)


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


def add_voronoi_regions(g, bounds):
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
        >>> add_voronoi_regions(g, bounds)
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
    for pointi in range(len(vor.point_region)):
        regioni = int(vor.point_region[pointi])  # convert to regular ints to may igraph happy
        r = vor.regions[regioni]

        if -1 not in r:
            # this region is fully defined, so add it
            g.vs[pointi]['cell'] = shapely.geometry.Polygon([vor.vertices[j] for j in r])
        else:
            print('Warning: unbounded region {}'.format(r))


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


def clip_cells(g, bounds):
    """ Clips all the cells in a graph to a bounding polygon.

        This algorithm is run in parallel because it may take a while if there
        are many vertices in the given graph.

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
        >>> add_voronoi_regions(g, bounds)
        >>> clip_cells(g, bounds)
    """
    # run this in parallel
    pool = multiprocessing.Pool()
    g.vs['cell'] = pool.map(_Clipper(bounds), g.vs['cell'])


def save_communities(g, n, filename, algorithm='label_propagation'):
    """ Finds a number of communities in a graph, and saves the shapes of those
        communities to a file.

        :param g: The graph to find communities in. Each node must have a cell
        attribute defining that node's area of influence.
        :param n: The number of times to run community detection.
        :param filename: The name of the file to shave the shapes to. Each
        polygon has a single property 'iteration' that contains the iteration
        during which that community was found.
        :param algorithm: The name of the communitiy detection algorithm to
        run. Should be one of: 'random_walk', 'eigenvector',
        'label_propagation', 'fast_greedy', or 'mulitilevel'. Default is
        'label_propagation'

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> add_voronoi_regions(g, bounds)
        >>> clip_cells(g, bounds)
        >>> iterations = 10
        >>> save_communities(g, iterations, 'test.shp')
        >>> save_communities(g, iterations, 'test_walk.shp', algorithm='random_walk')
    """
    communities = get_communities(g, n, algorithm)

    # Write the shapes to a file
    schema = {
        'geometry': 'Polygon',
        'properties': {'iteration': 'int'},
    }
    with fiona.open(filename, 'w', 'ESRI Shapefile', schema) as c:
        for i in range(n):
            for poly in communities[i]:
                c.write({'geometry': shapely.geometry.mapping(shapely.geometry.shape(poly)),
                         'properties': {'iteration': i}})


def get_communities(g, n, algorithm):
    """ Gets a number communities from a graph.

        :param g: The graph to find communities in. Each vertex must have a
        cell attribute representing that node's area of influence.
        :param n: The number of iterations of community detection to run
        :param algorithm: The name of the communitiy detection algorithm to
        run. Should be one of: 'random_walk', 'eigenvector',
        'label_propagation', 'fast_greedy', or 'mulitilevel'
        :return: list<list<Polygon>> Each list of polygons are the borders
        found in a single iteration.

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Read('test.graphml')
        >>> bounds = get_bounds(g)
        >>> add_voronoi_regions(g, bounds)
        >>> clip_cells(g, bounds)
        >>> communities = get_communities(g, 10, 'fast_greedy')
        >>> len(communities)
        10
    """
    communities = [[] for _ in range(n)]
    for iteration in range(n):
        ### find communities ###\
        algorithms = {'random_walk': lambda g: g.community_walktrap(weights='weight').as_clustering(),
                      'eigenvector': lambda g: g.community_leading_eigenvector(),
                      'label_propagation': lambda g: g.community_label_propagation(weights='weight'),
                      'fast_greedy': lambda g: g.community_fastgreedy(weights='weight').as_clustering(),
                      'multilevel': lambda g: g.community_multilevel(weights='weight')}
        comms = algorithms[algorithm](g)

        # find all of the cells in each community
        polys = [[] for _ in comms]  # polys : list<list<Polygon>>
        for i in range(len(comms)):
            for node_index in comms[i]:
                p = g.vs[node_index]['cell']
                if p.geom_type is "MultiPolygon":
                    polys[i].extend(p)
                else:
                    polys[i].append(p)
        # merge each list of polygons
        communities[iteration] = map(shapely.ops.cascaded_union, polys)

    return communities


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
        if p.geom_type == 'Polygon':
            p = [shapely.geometry.LineString(list(p.exterior.coords))]
        elif p.geom_type == 'MultiPolygon':
            p = [shapely.geometry.LineString(list(pi.exterior.coords)) for pi in p]
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
        print('{} / {}'.format(len(unique_borders), len(line_dict)))
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
                    if border.equals(other_border):
                        # these are the same border, ignore
                        continue
                    # check for intersection
                    i = border.intersection(other_border)
                    if i.length > 0.0:
                        print('adjusting borders')
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

    return unique_borders


def save_borders(communities_path, borders_path):
    """ Converts a community shape file to a border shape file.

        The community shape file should have a list of polygons defining the
        shapes of a community. The border file will contain a list of lines
        with weights corresponding to the number of times that border appeared
        in the community shape file.

        :param communities_path: The path to the community shape file
        :param borders_path: The path to the output border shape file

        Examples
        --------
        >>> save_borders('test_communities.shp', 'test_borders.shp')
    """
    borders = quantify_borders(communities_path)
    schema = {'geometry': 'MultiLineString',
              'properties': {'weight': 'int'}}
    with fiona.open(borders_path, 'w', 'ESRI Shapefile', schema) as c:
        for border, w in borders.iteritems():
            c.write({'geometry': shapely.geometry.mapping(shapely.geometry.shape(border)),
                     'properties': {'weight': w}})


def quantify_borders(shp_file):
    """ Takes the borders in the given shape file and find the number of times
        any segment appears.

        The resulting map contains line segments as keys, and the number of
        times that unique segment appears in the shapes in the given shape
        file as values.

        Takes string
        Returns map<shape, int>
    """
    temp_list = []
    with fiona.open(shp_file) as inp:
        for rec in inp:
            temp_list.append(shapely.geometry.asShape(rec['geometry']))

    line_dict = _collapse_duplicate_areas(temp_list)

    return _separate_borders(line_dict)


def border_plot(shp_file):
    # TODO: Make this work more generally
    m = Basemap(width=49000,
                height=38000,
                projection='lcc',
                resolution='h',
                lat_0=39.307556,
                lon_0=-76.600933)
    shp_info = m.readshapefile(shp_file, 'comm_bounds', drawbounds=False)
    colormap = plt.get_cmap('winter')
    for border, shape in zip(m.comm_bounds_info, m.comm_bounds):
        xx, yy = zip(*shape)
        m.plot(xx, yy, linewidth=4, color=colormap(border['weight'] / 8.0))

    m.drawcoastlines(linewidth=0.3)

    plt.show()


if __name__ == '__main__':
    print("Loading Graph")

    path = 'la_zip_distance_50k_1.6'
    iterations = 10
    g = igraph.Graph.Read('data/networks/{}.graphml'.format(path))

    print("Finding Cells")

    layout_position(g)
    bounds = get_bounds(g)
    add_voronoi_regions(g, bounds)

    print("Clipping Cells")

    clip_cells(g, bounds)

    print("Finding {} Communities".format(iterations))

    save_communities(g, iterations, 'output/communities/{}.shp'.format(path))

    print("Counting Borders")

    save_borders('output/communities/{}.shp'.format(path),
                 'output/borders/test.shp'.format(path))

    print("Plotting")

    lat, lon = (34.042988, -118.25145)
    m = Basemap(width=144000,
                height=144000,
                projection='lcc',
                resolution='h',
                lat_0=lat,
                lon_0=lon)
    shp_info = m.readshapefile('output/borders/test'.format(path), 'comm_bounds', drawbounds=False)
    colormap = plt.get_cmap('winter')
    max_weight = max([border['weight'] for border in m.comm_bounds_info])
    print(max_weight)
    for border, shape in zip(m.comm_bounds_info, m.comm_bounds):
        xx, yy = zip(*shape)
        m.plot(xx, yy, linewidth=4, color=colormap(border['weight'] / float(max_weight)))

    m.drawcoastlines(linewidth=1)

    plt.show()

