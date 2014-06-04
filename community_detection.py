__author__ = 'Tobin Yehle'

import igraph
import math
import shapely.ops
import shapely.geometry
import fiona
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def layout_position(g):
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]


def get_bounds(g, max_dimension=1600):
    xs = [float(x) for x in g.vs['longitude']]
    ys = [-float(y) for y in g.vs['latitude']]
    bounds = (0, 0, max(xs) - min(xs), max(ys) - min(ys))
    scale_factor = max_dimension / max(bounds)
    return [int(x*scale_factor) for x in bounds]


def find_clusters(g):
    comms = g.community_fastgreedy(weights='weight').as_clustering()
    return comms


def quantify_borders(shp_file):
    temp_list = []
    with fiona.open(shp_file) as inp:
        for rec in inp:
            temp_list.append(shapely.geometry.asShape(rec['geometry']))

    # Reduce all borders to unique Polygons
    line_dict = dict()
    while(len(temp_list) > 0):
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


    # Assign weights
    unique_borders = dict()
    done = False
    while(not done):
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


def border_plot(shp_file):
    m = Basemap(width = 49000, height = 38000,projection='lcc', resolution='h', lat_0 = 39.307556, lon_0 = -76.600933)
    shp_info = m.readshapefile(shp_file, 'comm_bounds', drawbounds = False)
    colormap = plt.get_cmap('winter')
    for border, shape in zip(m.comm_bounds_info, m.comm_bounds):
        xx,yy = zip(*shape)
        m.plot(xx,yy,linewidth = 4, color= colormap(border['weight'] / 8.0))

    m.drawcoastlines(linewidth = 0.3)

    plt.show()


if __name__ == '__main__':
    # # grab a graph from a file
    # # good: multilevel @ 0.88, leading eigenvector @ 0.85, walktrap @ 0.85
    # g = igraph.Graph.Read('output/la_zip_distance_0.8.graphml')
    # bounds = get_bounds(g)
    # layout_position(g)
    # g.vs['size'] = 8
    # c = find_clusters(g)
    # igraph.plot(c, mark_groups=True, bbox=bounds)

    borders = quantify_borders('comm_testing/comm_tester_double.shp')
    print(borders)
    # Write to Shapefile
    schema = {
        'geometry': 'MultiLineString',
        'properties': {'weight': 'int'},
    }

    with fiona.open('comm_testing/comm_quant_borders.shp', 'w', 'ESRI Shapefile', schema) as c:
        for border, w in borders.iteritems():
            c.write({'geometry': shapely.geometry.mapping(border),
                     'properties': {'weight': w}})

