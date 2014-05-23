"""
Created on Tue May 21 02:03 2014

@author: swhite
"""

from pymongo import *
import datetime

client = MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes
zips = db.zipcodes

def find_zips(field1, s1, field2 = None, s2 = None):
    """Return list of zipcodes (as strings) that are relevant to the given field.

    Arguments:
    field1, field2 -- string pertaining to a single field of the zips database
    s1, s2 -- instance of field as string, or a list of instances"""
    if type(s1) == str:
        s1 = [s1]
    if s2 != None:
        if type(s2) == str:
            s2 = [s2]
        return [str(i["zip"]) for i in zips.find({field1: {'$in': s1}, field2: {'$in': s2}})]
    return [str(i["zip"]) for i in zips.find({field1: {'$in': s1}})]
    

def crime_window(date1, date2, z_list = None, c_list = None):
    """Return all crimes within the given time window that ocurred in the given zipcode and were of type given.

    Arguments:
    date1 -- datetime object
    date2 -- datetime object
    z_list -- list of zipcodes as strings
    c_list -- list of crimes as strings"""
    if c_list == None and z_list == None: 
        # return time window for all zipcodes and crime types
        return crimes.find({"date": {"$gt": date1, "$lt": date2}})
    elif c_list == None:
        # return time window and zipcodes for all crimes
        return crimes.find({"date": {"$gte": date1, "$lte": date2}, "zipcode": {"$in": z_list}})
    elif z_list == None:
        # return time window and crimes for all zipcodes
        return crimes.find({"date": {"$gt": date1, "$lt": date2}, "type": {"$in": c_list}})
    else:
        # return crimes that satisfy all parameters
        return crimes.find({"date": {"$gt": date1, "$lt": date2}, "type": {"$in": c_list}, "zipcode": {"$in": z_list}})

def crime_window_v1(date1, date2, z_list):
    """Return all crimes within given time window that occurred within the given zipcodes.

    Arguments:
  date1 -- datetime object
  date2 -- datetime object such that date1 != date2
  z_list -- list of zipcodes as strings, note: list may be empty"""

    if len(z_list) == 0: 
      # if z_list is empty, return time window for all zipcodes
      return crimes.find({"date": {"$gt": date1, "$lt": date2}})
    else:
        return crimes.find({"date": {"$gt": date1, "$lt": date2}, "zipcode": {"$in": z_list}})
