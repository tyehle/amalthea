__author__ = 'tobin'

# import matplotlib
# matplotlib.use('Agg')  # this fixes issues when executing over ssh
import shapely.geometry
from shapely.ops import cascaded_union
import fiona
import matplotlib.pyplot as plt
from random import random
from matplotlib.collections import PatchCollection, PolyCollection
from descartes import PolygonPatch
import logging
from mpl_toolkits.basemap import Basemap
import community_detection

logger = logging.getLogger(__name__)


def get_region_fig(path, filename, region_type):
    """ Gets a figure with a plot of the regions of a network.

        :param path: The base path for the network
        :param filename: The filename of the network
        :return: The figure with the plot in it
    """
    cells = [shapely.geometry.shape(p['geometry']) for p in
             fiona.open('{}/regions/{}/{}.shp'.format(path, region_type, filename))]

    color_map = plt.get_cmap('Set1')
    fig = plt.figure()

    patches = []
    for c in cells:
        color = color_map(random())
        factory = lambda poly: PolygonPatch(poly, fc=color, ec='k', alpha=1, zorder=1)
        if c.geom_type is 'Polygon':
            patches.append(factory(c))
        elif c.geom_type is 'MultiPolygon':
            patches.extend([factory(p) for p in c])
        else:
            print("Warning: Ignoring unknown polygon type: {}".format(c.geom_type))
    fig.gca().add_collection(PatchCollection(patches, match_original=True))
    fig.gca().autoscale()
    return fig


def get_cluster_fig(g, clustering):
    """ Gets a figure with a plot of the regions of a network.

        :param g: The graph with cells in it
        :param clustering: The VertexClustering object to plot
        :return: The figure with the plot in it
    """
    cells = [cascaded_union([g.vs[i]['cell'] for i in nodes]) for nodes in clustering]

    color_map = plt.get_cmap('Set1')
    fig = plt.figure()

    patches = []
    for c in cells:
        color = color_map(random())
        factory = lambda poly: PolygonPatch(poly, fc=color, ec='k', alpha=1, zorder=1)
        if c.geom_type is 'Polygon':
            patches.append(factory(c))
        elif c.geom_type is 'MultiPolygon':
            patches.extend([factory(p) for p in c])
        else:
            print("Warning: Ignoring unknown polygon type: {}".format(c.geom_type))
    fig.gca().add_collection(PatchCollection(patches, match_original=True))
    fig.gca().autoscale()
    return fig


def get_adjacency_fig(path, filename, region_type):
    """ Gets a figure showing the adjacency network.
    """
    g = community_detection.load_network(path, filename)
    community_detection.add_regions(g, path, filename, region_type)
    adj = community_detection.get_adjacency_network(g, path, filename, region_type)

    fig, ax = plt.subplots()
    union = cascaded_union(g.vs['cell'])
    (left, bottom, right, top) = union.bounds
    m = get_map(left, right, top, bottom, ax)
    shade_regions_with_data(m, ax, path, filename, region_type)
    regions_path = '{}/regions/{}/{}'.format(path, region_type, filename)
    m.readshapefile(regions_path, 'regions', drawbounds=True, linewidth=.5, color='k', ax=ax)
    add_graph_to_map(adj, m)
    return fig


def get_border_fig(path, filename, region_type, algorithm, iterations):
    """ Plots the borders in a shapefile.

        :return: A figure with the borders
    """
    logger.info('Plotting Borders')

    border_path = '{}/borders/{}/{}/{}_{}'.format(path,
                                                  region_type,
                                                  algorithm,
                                                  filename,
                                                  iterations)

    # fig = plt.figure()
    fig, ax = plt.subplots()
    union = cascaded_union([shapely.geometry.shape(p['geometry']) for p in
                            fiona.open(border_path+'.shp')])
    (left, bottom, right, top) = union.bounds
    # assume we are in the northern hemisphere, west of the meridian
    m = get_map(left, right, top, bottom, ax)
    colormap = plt.get_cmap('gist_heat')

    # color the places we have data
    shade_regions_with_data(m, ax, path, filename, region_type)

    # draw the data
    m.readshapefile(border_path, 'comm_bounds', drawbounds=False)
    max_weight = max([border['weight'] for border in m.comm_bounds_info])
    for border, shape in zip(m.comm_bounds_info, m.comm_bounds):
        xx, yy = zip(*shape)
        # m.plot(xx, yy, linewidth=1, color=colormap(border['weight'] / float(max_weight)))
        m.plot(xx, yy, linewidth=1, alpha=border['weight'] / float(max_weight), color='purple')

    # TODO: Make the color bar actually work
    # m.colorbar()

    return fig


def get_map(left, right, top, bottom, ax, pad=.05):
    """ Gets a Basemap instance for the given area.

        The continents and oceans are filled in, and the coastline is drawn.

        :param ax: The axes to draw on.
        :param pad: The percentage of extra space on the edges of the map.
    """
    # assume bounds do not cross the antimeridian
    width = right - left
    height = top - bottom
    left -= pad * width
    right += pad * width
    bottom -= pad * height
    top += pad * height
    m = Basemap(resolution='h',
                projection='merc',
                llcrnrlon=left,
                llcrnrlat=bottom,
                urcrnrlon=right,
                urcrnrlat=top,
                ax=ax)
    m.drawmapboundary(color='k', linewidth=1.0, fill_color='#006699')
    m.fillcontinents(color='.6', zorder=0)
    m.drawcoastlines(linewidth=1)
    return m


def add_graph_to_map(g, m, c='r'):
    """ Plots a network on a map.
        :param g: The network to plot
        :param m: The map to plot on
        :param c: The color of the vertices of the network
    """
    # transform to plot coordinates
    xs, ys = m(g.vs['longitude'], g.vs['latitude'])
    # plot the edges
    for e in g.es:
        x = [xs[e.source], xs[e.target]]
        y = [ys[e.source], ys[e.target]]
        m.plot(x, y, 'k-', zorder=1)
    # plot the points next, so they go over the edges
    m.scatter(xs, ys, c=c, s=20, marker='o', zorder=2)


def shade_regions_with_data(m, ax, path, filename, region_type):
    """ Shades the all regions in a network on a map.

        This is used to show where we have data, and where we do not.
        :param m: The map to draw on
        :param ax: The axes to draw on
        :param path: The base path of the network
        :param filename: The filename of the network
        :param region_type: The region type to fill
    """
    regions_path = '{}/regions/{}/{}'.format(path, region_type, filename)
    m.readshapefile(regions_path, 'regions', drawbounds=False)

    # regions = m.regions
    regions = []
    for r in m.regions:
        if r not in regions:
            regions.append(r)

    with_data = PolyCollection(regions, edgecolors='none', facecolors=(1, 1, 1, .25))
    ax.add_collection(with_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _city = 'los_angeles'
    _distance = 1.6
    _node_type = 'crime'
    _path = 'data/{}/distance/{}/{}'.format(_city, _distance, _node_type)
    _filename = 'dec2010'
    _region_type = 'voronoi'
    _algorithm = 'label_propagation'
    _iterations = 30

    # _g = community_detection.load_network(_path, _filename)
    # community_detection.add_regions(_g, _path, _filename, _region_type)
    # _cs = community_detection.get_communities(_g, 1, _path, _filename, _algorithm)
    # _fig = get_cluster_fig(_g, _cs[0])

    # _fig = get_region_fig(_path, _filename, _region_type)
    # _fig = get_adjacency_fig(_path, _filename, _region_type)
    _fig = get_border_fig(_path, _filename, _region_type, _algorithm, _iterations)
    # _fig.set_size_inches(9, 9)
    plt.savefig('test.png', dpi=400)
    plt.show()