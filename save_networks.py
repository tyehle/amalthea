import datetime
import json
import network_creation
from windows import crime_window, str2date
import os.path
from math import ceil
import logging.config
import multithreading
import box_networks

logger = logging.getLogger(__name__)


def save_graph(g, file_path):
    path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(path):
        os.makedirs(path)
    g.write_graphml(file_path)


# def save_distance_zip_networks(cache=True):
#     city_zips = json.load(open('cities.json', 'r'))
#     cities = ['los_angeles', 'baltimore']
#     distances = [.1, .8, 1.6, 2.4, 3.2]
#
#     # TODO: Make this parallel if the network creation cannot be made parallel
#
#     for city in cities:
#         # grab the data we will use
#         data = crime_window(zipcodes=city_zips[city],
#                             start_date=datetime(2010, 12, 30),
#                             end_date=datetime(2011, 1, 1))
#         filename = '30dec2010'
#         for d in distances:
#             print('{} {}'.format(city, d))
#             base_path = 'data/{}/distance/{}'.format(city, d)
#             crime_path = '{}/crime/networks/{}.graphml'.format(base_path, filename)
#             zip_path = '{}/zip/networks/{}.graphml'.format(base_path, filename)
#             # if the graph already exists, do not make a new one
#             if os.path.exists(zip_path):
#                 continue
#             if not os.path.exists(crime_path):
#                 # no crime network found
#                 if cache:
#                     # if we want to cache the crime network, make one
#                     g = network_creation.distance_crime_graph(data, d)
#                     save_graph(g, crime_path)
#                     # reduce and save the network
#                     network_creation.reduce_to_zip_graph(g)
#                     save_graph(g, zip_path)
#                 else:
#                     # do not cache (probably for memory reasons)
#                     g = network_creation.distance_zip_graph(data, d)
#                     save_graph(g, zip_path)
#             else:
#                 # the crime network already exists, so grab it
#                 g = igraph.Graph.Read(crime_path)
#                 # reduce the network
#                 network_creation.reduce_to_zip_graph(g)
#                 save_graph(g, zip_path)


def date_string(d):
    return d.strftime('%Y-%m-%d')


def get_crime_name(crime_types):
    if crime_types is None:
        return 'all'
    else:
        return '-'.join([t.lower() for t in crime_types])


def save_dynamic_distance_delta_graph(initial, final, delta_name, area_name,
                                      distance, node_type, crime_types=None):
    """ Creates graphs per each unit of delta time given a window of crime.

        A crime window is created using the relevant given paremeters. For each
        increment of delta between time initial and time final, a graph of the
        crime window for relevant times is saved as a .graphml file using a
        unique file name.

        Parameters
        ----------
        initial: datetime.datetime
            Initial time of crimes used to retrieve crime window.
        final: datetime.datetime
            Final time of crimes used to retrieve crime window.
        delta: datetime.timedelta
            Time difference of interest.
        area_name:
            String indicating the name of the city find crimes in.
        distance: float
            Maximum distance between linked crimes.
        node_type: String
            What each node in the network represents. Should be one of 'zip' or 'crime'.
        crime_types: list
            An optional additional parameter passed to `crime_window`

        Returns
        -------
        .graphml
            Multiple .graphml files with unique names indicative of the time
            delta at hand.

        Examples
        --------
        >>> from datetime import datetime, timedelta
        >>> initial = datetime(2010, 1, 1)
        >>> final = datetime(2010, 1, 8)
        >>> delta = timedelta(days=1)
        >>> save_dynamic_distance_graph(initial, final, delta, 'baltimore', 1.6, 'zip')
    """
    path = 'data/{}/{}/distance/{}/{}'.format(area_name, get_crime_name(crime_types),
                                              distance, node_type)
    zipcodes = json.load(open('cities.json', 'r'))[area_name]
    if delta_name == 'week':
        delta = datetime.timedelta(days=7)
    elif delta_name == 'day':
        delta = datetime.timedelta(days=1)
    else:
        raise NotImplementedError('Unrecognized delta name: {}'.format(delta_name))
    # Create crime window
    w = crime_window(start_date=initial, end_date=final, zipcodes=zipcodes, crime_types=crime_types)
    logger.info('{} crimes found'.format(len(w)))
    # Calculate number of increments
    increment = int(ceil((final - initial).total_seconds()/delta.total_seconds()))
    t = initial
    # Create graph for each increment
    for c in range(increment):
        logger.info('Creating graph {} of {}'.format(c, increment))
        cur_t = t + delta
        # Filter relevant crimes
        c_relevant = []
        for crime in w:
            d = str2date(crime['date'])
            if t <= d < cur_t:
                c_relevant.append(crime)
        g = network_creation.distance_graph(c_relevant, distance, node_type)
        # g.vs['y'] = [-x for x in g.vs['latitude']]
        # g.vs['x'] = [x for x in g.vs['longitude']]
        # Size node proportional to node betweenness
        # m = max([g.betweenness(i) for i in g.vs])
        # g.vs['size'] = [((g.betweenness(i)/m) * 12) + 5 for i in g.vs]
        save_graph(g, '{}/networks/{}_{}.graphml'.format(path, delta_name, date_string(t)))
        #igraph.plot(g)
        t = cur_t


def save_dynamic_distance_month_graph(years, area_name, distance, node_type, crime_types=None):
    start_times = map(lambda args: datetime.datetime(**args),
                      multithreading.combinations(month=range(1, 13), year=years, day=[1]))
    # add the end point of the final network to the list
    start_times.append(datetime.datetime(year=years[-1] + 1, month=1, day=1))


    path = 'data/{}/{}/distance/{}/{}'.format(area_name, get_crime_name(crime_types), distance, node_type)
    zipcodes = json.load(open('cities.json', 'r'))[area_name]
    i = 0
    while i < len(start_times) - 1:
        data = crime_window(start_times[i], start_times[i+1], zipcodes, crime_types)
        logger.debug('{} crimes found for {}'.format(len(data), start_times[i]))
        g = network_creation.distance_graph(data, distance, node_type)
        save_graph(g, '{}/networks/{}_{}.graphml'.format(path, 'month', date_string(start_times[i])))
        i += 1


def save_dynamic_distance_year_graph(years, area_name, distance, node_type, crime_types=None):
    start_times = map(lambda args: datetime.datetime(**args),
                      multithreading.combinations(month=[1], year=years, day=[1]))
    # add the end point of the final network to the list
    start_times.append(datetime.datetime(year=years[-1]+1, month=1, day=1))
    path = 'data/{}/{}/distance/{}/{}'.format(area_name, get_crime_name(crime_types), distance, node_type)
    box_networks._multiprocess = False  # this is very bad, do not multiprocess!
    zipcodes = json.load(open('cities.json', 'r'))[area_name]

    if crime_types is None:
        limits = {'zipcode': {'$in': zipcodes}}
    else:
        limits = {'zipcode': {'$in': zipcodes}, 'type': {'$in': crime_types}}

    i = 0
    while i < len(start_times) - 1:
        limits['date'] = {'$gte': start_times[i], '$lt': start_times[i+1]}
        g = box_networks.distance_crime_network_by_box(distance, area_name, limits=limits)
        save_graph(g, '{}/networks/{}_{}.graphml'.format(path, 'year', date_string(start_times[i])))
        i += 1


if __name__ == '__main__':
    """ Create dynamic graphs for all of 2010 by days and weeks for baltimore
        and los angeles for multiple distance networks."""
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))

    todo = 'month'

    areas = ['baltimore', 'los_angeles', 'miami']
    distances = [0.1, 0.8, 1.6, 2.4, 3.2]
    node_types = ['crime']
    crime_types = [None, ['Theft'], ['Burglary'], ['Assault']]

    logger.info('Starting')
    if todo == 'month':
        params = multithreading.combinations(years=[range(2007, 2011)],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=crime_types)
        multithreading.map_kwargs(save_dynamic_distance_month_graph, params)
    elif todo == 'week':
        params = multithreading.combinations(initial=[datetime.datetime(2007, 1, 1)],
                                             final=[datetime.datetime(2011, 1, 1)],
                                             delta_name=['week'],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=crime_types)
        logger.info('Generating {} dynamic networks'.format(len(params)))
        multithreading.map_kwargs(save_dynamic_distance_delta_graph, params)
        # map(lambda args: save_dynamic_distance_graph(**args), params)
    elif todo == 'year':
        params = multithreading.combinations(years=[range(2007, 2011)],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=crime_types)
        multithreading.map_kwargs(save_dynamic_distance_year_graph, params)
    logger.info('Done!')
