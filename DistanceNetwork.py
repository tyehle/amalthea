"""
Created on Tue May 20 15:13 2014

@author: swhite
"""

from pymongo import *
import igraph
import datetime
from math import radians, cos, sin, asin, sqrt
import windows

client = MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes

def v_distance(lat1, lon1, lat2, lon2):
    """Return distance between the two given locations in miles.
    Formula from: http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

    Arguments:
    lat1, lon1, lat2, lon2- float indicative of latitude/longitude location"""
    
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a)) 
    R = 3961  # Earth's radius in miles
    return R * c
    
def create_dNetwork(crime_window, dist):
    """Return graph of given crime data where edges arise when vertices are less than dist miles from each other.

    Arguments:
    crime_window -- cursor object of crime data
    dist -- float indcative of desired distance between two points such that a edge is present"""

    g = igraph.Graph()
    # Traverse crimes in window, for each crime create a vertex
    for crime in crime_window:
        g.add_vertices(1)
        c = g.vcount() - 1
        g.vs[c]["date"] = crime["date"]   # Rewrite using **
        g.vs[c]["zipcode"] = crime["zipcode"]
        g.vs[c]["latitude"] = float(crime["latitude"])
        g.vs[c]["longitude"] = float(crime["longitude"])
        g.vs[c]["address"] = crime["address"]
        g.vs[c]["type"] = crime["type"]
        # Connect vertices within dist miles of each other
        if c > 0:
            for v in range(c):
                lat1 = g.vs[c]["latitude"]
                lon1 = g.vs[c]["longitude"]
                lat2 = g.vs[v]["latitude"]
                lon2 = g.vs[v]["longitude"]
                d = v_distance(lat1, lon1, lat2, lon2)
                if d <= dist:
                    g.add_edge(c, v, dist = d)
    return g

# Testing code:
# z = windows.find_zips('city', ['Orlando'])
# w = windows.crime_window(datetime.datetime(2009, 1, 1), datetime.datetime(2009, 8, 1), z_list = z, c_list = ['Robbery'])
# g = create_dNetwork(w, 2)
# print g.summary()
# g.vs['size'] = 4
# g.vs['x'] = g.vs['latitude']
# g.vs['y'] = g.vs['longitude']
# # g.es['edge_width'] = g.es[dist]
# # g.vs['label'] = ["(%.2f,%.2f)"%(i,j) for i,j in zip(g.vs['x'],g.vs['y'])]
# igraph.plot(g)

