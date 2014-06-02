"""
Created on Tue May 21 02:03 2014

@author: swhite
"""

from pymongo import *

client = MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes
zips = db.zipcodes

def crime_window(start_date=None,
                 end_date=None,
                 cities=None,
                 states=None,
                 zipcodes=None,
                 crime_types=None,
                 max_size=None):
    """ Return all crimes within the given limits.

        Parameters
        ----------
        start_date, end_date: datetime
            The output will contain only crimes within the given window.
        cities, states: list
            The output will contain only crimes in the given cities AND the
            given states.
        zipcodes: list
            Output will only contain crimes in the given zipcodes. This list
            will override any constraints specified by the list of cities and
            states.
        crime_types: list
            The results will contain only crimes of the given type.
        max_size: int
           Limit the total number of crimes fetched from the database.

        Returns
        -------
        crimes : list
            A list of crimes, where each crimes is a dictionary

        Examples
        --------
        >>> crimes = crime_window(max_size=2)
        >>> len(crimes)
        2

        >>> from datetime import datetime
        >>> crimes = crime_window(end_date=datetime(2009, 2, 15), start_date=datetime(2009, 2, 13))
        >>> len(crimes)
        21669
    """

    limits = dict()

    if start_date is not None:
        limits['date'] = {'$gte': start_date}

    if end_date is not None:
        if 'date' in limits:
            limits['date']['$lte'] = end_date
        else:
            limits['date'] = {'$lte': end_date}

    zips_limits = dict()
    if cities is not None:
        zips_limits['city'] = {'$in': cities}
    if states is not None:
        zips_limits['state'] = {'$in': states}

    if zipcodes is not None:
        if cities is not None or states is not None:
            print("Warning: Overwriting city and state limitations!")
            zips_limits = dict()

        limits['zipcode'] = {'$in': zipcodes}

    if len(zips_limits) is not 0:
        limits['zipcode'] = {'$in': [str(z['zip']) for z in zips.find(zips_limits)]}

    if crime_types is not None:
        limits['type'] = {'$in': crime_types}

    if max_size is None:
        max_size = 0

    c_window = [c for c in crimes.find(limits, limit=max_size)]

    for c in c_window:
        c['description'] = c['description'].decode('utf-8')
        c['address']  = c['address'].deocde('utf8')
        c['latitude'] = float(c['latitude'])
        c['longitude'] = float(c['longitude'])

    return c_window
