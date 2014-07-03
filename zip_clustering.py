import os
import csv
import json
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import pdist, squareform 
from scipy.cluster import hierarchy
import fiona
from shapely.geometry import mapping, asShape
import igraph
import plotting

_client = MongoClient('163.118.78.22', 27017)
_db = _client['crimes_test']
_geometry = _db.geometry

def retrieve_census(area_name, 
                    file_names = ['aff_download/ACS_12_5YR_DP02_with_ann.csv', 
                                 'aff_download/ACS_12_5YR_DP03_with_ann.csv', 
                                 'aff_download/ACS_12_5YR_DP04_with_ann.csv'], 
                    cols = [[33, 237, 241, 321], 
                           [37, 247, 297, 289, 513], 
                           [13, 185]]):
    """ Reads census data from files for variables of interest. Outputs a dictionary of features by zip code and a dictionary of variables.
    """
    features = dict()
    var_names = []
    f = 'data/{}/census/features.json'.format(area_name)
    if os.path.exists(f):
        features = json.load(open('data/{}/census/features.json'.format(area_name), 'r'))
        var_names = json.load(open('data/{}/census/var_names.json'.format(area_name), 'r'))
    else:
        c = json.load(open('cities.json'))
        z_list = c[area_name]
        if len(var_names) == 0:
            count = 0
            with open(file_names[0]) as data:
                traverse = csv.reader(data)
                for row in traverse:
                    count += 1
                    if count > 2:
                        if row[1] in z_list:
                            features[row[1]] = []
        for ind, f in enumerate(file_names):
            with open(f) as data:
                zip_data = csv.reader(data)
                count = 0
                for row in zip_data:
                    count += 1
                    if count == 1:
                        continue
                    # Record the variable names
                    if count == 2:
                        var_names = var_names + [row[i] for i in cols[ind]]
                    # Add zipcode data to dictionary
                    else:
                        if z_list == None or row[1] in z_list:
                            for i in cols[ind]:
                                try:
                                    features[row[1]].append(float(row[i]))
                                except ValueError:
                                    features[row[1]].append(float(0.0001))
        var_names = dict(zip(range(len(var_names)), var_names))
        json.dump(features, open('data/{}/census/features.json'.format(area_name), 'w'))
        json.dump(var_names, open('data/{}/census/var_names.json'.format(area_name), 'w'))
    return (features, var_names)


def cluster_zips(area_features, linkage, t, return_dist = False):
    """ Clusters zip codes using a hierachial method with euclidean distance and the inputted feature vector.
    """
    if type(area_features) == str:
        features = json.load(open('data/{}/census/features.json'.format(area_features), 'r'))
    elif type(area_features) == dict:
        features = area_features
    feat = []
    for i in features.values():
        feat.append(i)
    y = pdist(np.matrix(feat), 'euclidean')
    if linkage == 'single':
        Z = hierarchy.single(y)
    elif linkage == 'average':
        Z = hierarchy.average(y)
    elif linkage == 'complete':
        Z = hierarchy.complete(y)
    f = hierarchy.fcluster(Z, criterion = 'distance', t = t)
    if return_dist == True:
        return (squareform(y), f)
    else:
        return f
    

def census_cluster_graph(area_name, linkage, t):
    # Open the feature and variables names dictionaries for the respective area
    features = json.load(open('data/{}/census/features.json'.format(area_name), 'r'))
    # var_names = json.load(open('data/{}/census/var_names.json'.format(area_name), 'r'))
    # Cluster the zip codes according the specified linkage and threshold
    y, Z = cluster_zips(features, linkage, t, return_dist=True)
    # Create dictionary of average statistics per cluster
    # zd = dict(zip([int(cluster) for cluster in set(Z)],[dict() for dictionary in range(len(set(Z)))]))
    # for k in zd.keys():
    #     for v in var_names.values():
    #         zd[k][v] = 0
    # Create a graph of the zip codes 
    g = igraph.Graph(len(features.keys()))
    g.vs['zipcode'] = features.keys()

    # Traverse the zip code nodes assigning edges to nodes with borders
    zip_list = []
    for node in g.vs: 
        # Load zip code's shape file
        geom = asShape(_geometry.find_one({'zip': node['zipcode']})['geometry'])
        zip_list.append(geom)
        # Add relevant attributes to node
        try:
            node['longitude'] = sum([geom.bounds[i]  for i in range(len(geom.bounds)) if i % 2 == 0]) * 0.5
            node['latitude'] = sum([geom.bounds[i]  for i in range(len(geom.bounds)) if i % 2 != 0]) * 0.5
            node['cluster'] = int(Z[node.index])
        except TypeError:
            print geom.bounds, type(geom.bounds)
        # Contribute zip code's statistics to cluster in dictionary
        # for i, k in enumerate(var_names.keys()):
        #         zd[int(Z[node.index])][var_names[str(i)]] += features[node['zipcode']][i]
        # Investigate relationships between current node and previously investigated nodes
        for i in range(node.index):
            # Add edge if zip codes are adjacent and are in different borders
            if zip_list[i].intersection(zip_list[node.index]).length > 0:
                if Z[i] != Z[node.index]:
                        # Quanitfy edge weight
                        c1 = [c for c in range(len(Z)) if Z[c] == Z[i]]
                        c2 = [c for c in range(len(Z)) if Z[c] == Z[node.index]]
                        m = 0
                        # Find complete distance between clusters
                        for blah in c1:
                            for blah2 in c2:
                                if y[blah][blah2] > m:
                                    m = y[blah][blah2]
                        # Add edge to graph with weight indicative of cluster distance
                        g.add_edge(i, node.index, weight = float(m))
                else:
                    g.add_edge(i, node.index, weight = 0.0)
    if len(g.es) > 0:
        # Normalize edge weights against maximum cluster distance
        n = 0.0
        for abc in range(len(Z)):
            for xyz in range(len(Z)):
                if y[abc][xyz] > n:
                    n = y[abc][xyz]
        n = float(n)
        if n > 0.0:
            g.es['weight'] = [i/n for i in g.es['weight']]
    g.write_graphml('data/{}/census/hierarchical/{}/{}_{}.graphml'.format(area_name, linkage, linkage, t))
    # for k in zd.keys():
    #     l = len([1 for c in Z if c == k])
    #     for stat in range(len(zd[k].values())):
    #         zd[k][zd[k].keys()[stat]] /= l 
    # json.dump(zd, open('data/{}/census/hierarchical/{}/{}_{}.json'.format(area_name, linkage, linkage, t), 'w'))


def census_cluster_plot(area_name, linkage, t):
    features = json.load(open('data/{}/census/features.json'.format(area_name), 'r'))
    schema = {'geometry': 'MultiPolygon',
              'properties': {'zip': 'str',
                            str(t): 'int'} }
    Z = cluster_zips(features, linkage, t)
    with fiona.open('data/{}/census/hierarchical/{}/{}_{}.shp'.format(area_name, linkage, linkage, t), 'w', 'ESRI Shapefile', schema) as c:
                        for i, zipc in enumerate(features.keys()):
                            poly = _geometry.find_one({'zip': zipc})['geometry']
                            c.write({
                                'geometry': mapping(asShape(poly)),
                                'properties': {'zip': zipc, 
                                              str(t): int(Z[i])}
                                               })
    # plotting.get_census_fig(area_name, linkage, t)


if __name__ == '__main__':
    test_iterations = 6
    cities = ['baltimore', 'los_angeles', 'miami']
    trials = [('single', 200000), ('complete', 200000), ('average', 200000)]
    for area in cities:
        for s in trials:
            for i in range(6):
                census_cluster_graph(area, s[0], int(s[1] * .5**i))
                # census_cluster_plot(area, s[0], int(s[1] * .5**i))

