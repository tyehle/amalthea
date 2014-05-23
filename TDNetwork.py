"""
Created on Tue May 20 15:13 2014

@author: swhite
"""

from pymongo import *
import igraph
import datetime
from math import radians, cos, sin, asin, sqrt
import windows
import DistanceNetwork

client = MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes

def create_tdNetwork(crime_window, dist, tim):
    """Return graph of given crime data where edges arise when vertices are 
    less than dist miles from each other and less than time minutes apart.

    Arguments:
    crime_window -- cursor object of crime data
    dist -- float indcative of desired distance in miles between two crimes such that a edge is present
    tim -- float indicative of time in minutes between two crimes such that a edge is present"""
    g = igraph.Graph()
    # Traverse crimes in window, for each crime create a vertex
    for crime in crime_window:
        g.add_vertices(1)
        c = g.vcount() - 1
        g.vs[c]["date"] = crime["date"]   # Rewrite using **
        g.vs[c]["zipcode"] = crime["zipcode"]
        g.vs[c]["latitude"] = float(crime["latitude"])
        g.vs[c]["longitude"] = float(crime["longitude"])
        # g.vs[c]["address"] = crime["address"]
        g.vs[c]["type"] = crime["type"]
        # Connect vertices within dist miles and time minutes of each other
        if c > 0:
            for v in range(c):
                d_delta = DistanceNetwork.v_distance(g.vs[c]["latitude"], g.vs[c]["longitude"], g.vs[v]["latitude"], g.vs[v]["longitude"])
                t_delta = abs((g.vs[c]["date"] - g.vs[v]["date"]).total_seconds()/60)
                if d_delta <= dist and t_delta <= tim:
                    g.add_edge(c, v)
    crime_window.rewind()
    return g


# z = windows.find_zips('state', 'FL', 'city', ['Miami'])
# w = windows.crime_window(datetime.datetime(2008, 1, 1), datetime.datetime(2008, 8, 1), z_list = z, c_list = ['Robbery'])
# g = create_tdNetwork(w, 2, 24*60)
# print g.summary()
# g.vs['size'] = 4
# #g.vs['label'] = ["(%.2f,%.2f)"%(i,j) for i,j in zip(g.vs['latitude'],g.vs['longitude'])]
# g.vs['y'] = [-x for x in g.vs['latitude']]
# g.vs['x'] = [x for x in g.vs['longitude']]
# igraph.plot(g)
# # i = 1
# # for x in g.vs['label']:
# #     print str(x.replace('(','').replace(')','') + ',' + 'point ' + str(i))
# #     i += 1