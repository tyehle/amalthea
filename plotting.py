__author__ = 'tobin'

# import matplotlib
# matplotlib.use('Agg')
import shapely.geometry
from shapely.ops import cascaded_union
import fiona
import matplotlib.pyplot as plt
from random import random
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import logging
from mpl_toolkits.basemap import Basemap

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


def get_border_fig(border_path, pad=.05):
    """ Plots the borders in a shapefile.

        :param border_path: The path to the shape file containing the borders
        to plot.
        :param pad: The percentage of the width to pad the edge of the map.
        :return: A figure with the borders
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
        # m.plot(xx, yy, linewidth=1, alpha=border['weight'] / float(max_weight), color='blue')

    m.drawcoastlines(linewidth=1)

    # TODO: Make the color bar actually work
    # m.colorbar()

    return fig

if __name__ == "__main__":
    _path = 'data/baltimore/distance/1.6/zip'
    _filename = 'dec2010'
    _fig = get_region_fig(_path, _filename, 'zip')
    _fig.set_size_inches(16, 9)
    plt.savefig('test.png', dpi=100)
    plt.show()