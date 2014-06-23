import os
import csv
import json
import numpy as np
from pymongo import MongoClient
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy

_client = MongoClient('163.118.78.22', 27017)
_db = _client['crimes_test']
_geometry = _db.geometry


def retrieve_census(city, 
                    file_names = ['ACS_12_5YR_DP02_with_ann.csv', 
                                 'ACS_12_5YR_DP03_with_ann.csv', 
                                 'ACS_12_5YR_DP04_with_ann.csv',
                                 'ACS_12_5YR_DP05_with_ann.csv'], 
                    cols = [[33, 237, 241, 321], 
                           [37, 247, 297, 289, 513], 
                           [13, 185],
                           [3]], 
                    features = dict(), 
                    var_names = []):
    """ Reads census data from files for variables of interest. Outputs a dictionary of features by zip code and a dictionary of variables.
    """
    dir = os.path.abspath('data/{}/census/features.json'.format(city))
    if os.path.exists(dir):
        features = json.load(open('data/baltimore/census/features.json', 'r'))
        var_names = json.load(open('data/baltimore/census/var_names.json', 'r'))
    else:
        c = json.load(open('cities.json'))
        z_list = c[city]
        if len(var_names) == 0:
            count = 0
            os.chdir('/home/swhite/amalthea/aff_download')
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
        json.dump(features, open('/home/swhite/amalthea/data/{}/census/features.json'.format(city), 'w'))
        json.dump(var_names, open('/home/swhite/amalthea/data/{}/census/var_names.json'.format(city), 'w'))
    return (features, var_names)


def cluster_zips(features, linkage, t):
    """ Clusters zip codes using a hierachial method with euclidean distance and the inputted feature vector.
    """
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
    return hierarchy.fcluster(Z, criterion = 'distance', t = t)
    




if __name__ == '__main__':
    import plotting
    import fiona
    from shapely.geometry import mapping, asShape

    os.chdir('/home/swhite/amalthea')
    cities = ['baltimore', 'los_angeles']
    results = []
    for i in cities:
        results.append(retrieve_census(i))
    schema = {'geometry': 'MultiPolygon',
              'properties': {'zip': 'str', 
                            'L1': 'int',
                            'L2': 'int',
                            'L3': 'int',
                            'L4': 'int',
                            'L5': 'int',
                            'L6': 'int'} }
    for i, city in enumerate(results):
        trials = [('single', 50000), ('complete', 50000), ('average', 50000)]
        levels = []
        for s in trials:
            os.chdir('/home/swhite/amalthea/data/{}/census/'.format(cities[i]))
            for i in range(6): 
                levels.append(cluster_zips(city[0], s[0], (s[1] * .5**(i+1))))
                if len(levels) == 6:
                    # Save clusters as shapefiles
                    with fiona.open('{}.shp'.format(s[0]), 'w', 'ESRI Shapefile', schema) as c:
                        for i, zipc in enumerate(city[0].keys()):
                            poly = _geometry.find_one({'zip': zipc})['geometry']
                            c.write({
                                'geometry': mapping(asShape(poly)),
                                'properties': {'zip': zipc, 
                                              'L1': int(levels[0][i]),
                                              'L2': int(levels[1][i]),
                                              'L3': int(levels[2][i]),
                                              'L4': int(levels[3][i]),
                                              'L5': int(levels[4][i]),
                                              'L6': int(levels[5][i])} })
                    levs = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
                    for l in levs:
                        plotting.get_census_fig(cities[i], s[0], l)
                    levels = []



