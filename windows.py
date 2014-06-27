from pymongo import MongoClient
from shapely.geometry import asShape, Polygon
from unidecode import unidecode
from datetime import datetime

client = MongoClient('163.118.78.22', 27017)
db = client['crimes']
crimes = db.crimes
geom = db.geometry

_datefmt = '%Y-%m-%d %H:%M:%S'


def str2date(s):
    return datetime.strptime(s, _datefmt)


def date2str(d):
    return d.strftime(_datefmt)


def crime_window(start_date=None,
                 end_date=None,
                 zipcodes=None,
                 crime_types=None,
                 max_size=None):
    """ Return all crimes within the given limits.

        Parameters
        ----------
        start_date, end_date: datetime
            The output will contain only crimes within the given window.
        zipcodes: list
            Output will only contain crimes in the given zipcodes.
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
            limits['date']['$lt'] = end_date
        else:
            limits['date'] = {'$lt': end_date}

    if zipcodes is not None:
        limits['zipcode'] = {'$in': zipcodes}

    if crime_types is not None:
        limits['type'] = {'$in': crime_types}

    if max_size is None:
        max_size = 0

    return normalize_data(crimes.find(limits, limit=max_size))


def normalize_data(crime_list):
    """ Normalizes the format of crimes from the database

        :param crime_list: A list or Cursor containing crimes from the database
        :return: A list containing the normalized data
    """
    # convert keys to strings
    data = [{str(k): v for k, v in data.iteritems()} for data in crime_list]

    for c in data:
        c['description'] = unidecode(c['description'])
        c['address'] = unidecode(c['address'])
        c['latitude'] = float(c['latitude'])
        c['longitude'] = float(c['longitude'])
        c['date'] = date2str(c['date'])

    return data


def zip_box(minlat, minlon, maxlat, maxlon):
    # Make a box
    box = Polygon([[minlon, minlat], [maxlon, minlat], [maxlon, maxlat], [minlon, maxlat]])
    # Traverse each zipcode and see if within/intersects box
    z_list = [z['zip'] for z in geom.find() if asShape(z['geometry']).intersects(box)]
    return z_list
