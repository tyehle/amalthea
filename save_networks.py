import datetime
import json
import network_creation
from windows import crime_window, str2date
import os.path
from math import ceil
import logging.config
import multithreading
import box_networks
import igraph
import glob

logger = logging.getLogger(__name__)


def save_graph(g, file_path):
    """ Saves a graph, creating the directory first if it does not exist.

        :param g: The graph to save.
        :param file_path: The path to write the graph to.

        Examples
        --------
        >>> import igraph
        >>> g = igraph.Graph.Full(7)
        >>> path = 'data/testing/test.graphml'
        >>> save_graph(g, path)
        >>> os.path.exists(path)
        True
    """
    path = os.path.dirname(os.path.abspath(file_path))
    if not os.path.exists(path):
        os.makedirs(path)
    g.write_graphml(file_path)


def collapse_all_to_zip():
    """ Creates zip code networks out of all the crimes networks available.

        Will not overwrite previously existing zip code networks.

        Examples
        --------
        >>> collapse_all_to_zip()
    """
    networks = glob.glob('data/*/*/distance/*/crime/networks/*.graphml')
    logger.info('{} networks found'.format(len(networks)))
    parts = [name.split('/') for name in networks]
    paths = [(os.path.join(*p[:5]), p[-1]) for p in parts]

    for base_path, filename in paths:
        logger.debug('{} : {}'.format(base_path, filename))
        # base_path = 'data/{}/distance/{}'.format(city, d)
        crime_path = '{}/crime/networks/{}'.format(base_path, filename)
        zip_path = '{}/zip/networks/{}'.format(base_path, filename)
        # if the graph already exists, do not make a new one
        if os.path.exists(zip_path):
            logger.debug('Network exists, skipping')
            continue
        elif os.path.exists(crime_path):
            logger.info('Collapsing {}'.format(crime_path))
            # the crime network already exists, so grab it
            try:
                g = igraph.Graph.Read(crime_path)
            except igraph.InternalError as e:
                logger.error('Failed to read {}, skipping'.format(crime_path))
                logger.exception(e)
                continue
            # only try to reduce the network if there are crimes in it
            if g.vcount() > 0:
                # reduce the network
                network_creation.reduce_to_zip_graph(g)
            save_graph(g, zip_path)
        else:
            raise RuntimeError('Crime network not found {}'.format(crime_path))


def date_string(d):
    """ Gets a string representing a date, but not time.

        This is used to generate readable dates for use in file names.

        :param d: A datetime object
        :return: A string representing the date

        Examples
        --------
        >>> from datetime import datetime
        >>> date_string(datetime(2010, 12, 14, 5, 31, 02))
        '2010-12-14'
    """
    return d.strftime('%Y-%m-%d')


def get_crime_name(crime_types):
    """ Converts a list of crime types into a readable file name.

        `None` is converted to 'all'

        :param crime_types: The list of crime types. Each type is a string.
        :return: A single string representing the list of types.

        Examples
        --------
        >>> get_crime_name(None)
        'all'
        >>> get_crime_name(['Theft', 'Burglary'])
        'theft-burglary'
    """
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
        >>> save_dynamic_distance_delta_graph(initial, final, delta, 'baltimore', 1.6, 'zip')
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
    # Calculate number of increments
    increment = int(ceil((final - initial).total_seconds()/delta.total_seconds()))

    # make sure we need to build these networks
    get_network_path = lambda start: '{}/networks/{}_{}.graphml'.format(path,
                                                                        delta_name,
                                                                        date_string(start))
    paths = [get_network_path(initial + delta*i) for i in range(increment)]
    all_exist = reduce(lambda all_good, p: all_good and os.path.exists(p), paths, True)
    if all_exist:
        return

    # Create crime window
    w = crime_window(start_date=initial, end_date=final, zipcodes=zipcodes, crime_types=crime_types)
    logger.info('{} crimes found'.format(len(w)))

    t = initial
    # Create graph for each increment
    for c in range(increment):
        logger.info('Creating graph {} of {}'.format(c, increment))
        cur_t = t + delta
        network_path = get_network_path(t)
        if not os.path.exists(network_path):
            # Filter crimes in this window
            c_relevant = filter(lambda crime: t <= str2date(crime['date']) < cur_t, w)
            g = network_creation.distance_graph(c_relevant, distance, node_type)
            save_graph(g, network_path)
        else:
            logger.debug('Network exists, skipping')
        t = cur_t


def save_dynamic_distance_month_graph(years, area_name, distance, node_type, crime_types=None):
    """ Saves a number of networks, where each network represents a month.

        Each network is saved to disk at a location that matches the parameters
        used to construct it.

        :param years: The time frame of networks to create. Should be a list of
        integers.
        :param area_name: The name of the area to search for crimes. This
        should be a valid entry in cities.json
        :param distance: The maximum distance between two connected crimes.
        :param node_type: What each node represents. The is passed on to
        network_creation.distance_graph.
        :param crime_types: The types of crimes to include in the network.

        Examples
        --------
        >>> save_dynamic_distance_month_graph([2008, 2009], 'miami', 1.6, 'crime', ['Theft'])
    """
    start_times = map(lambda args: datetime.datetime(**args),
                      multithreading.combinations(month=range(1, 13), year=years, day=[1]))
    # add the end point of the final network to the list
    start_times.append(datetime.datetime(year=years[-1] + 1, month=1, day=1))

    path = 'data/{}/{}/distance/{}/{}'.format(area_name, get_crime_name(crime_types), distance, node_type)
    zipcodes = json.load(open('cities.json', 'r'))[area_name]
    i = 0
    while i < len(start_times) - 1:
        network_path = '{}/networks/{}_{}.graphml'.format(path, 'month', date_string(start_times[i]))
        if not os.path.exists(network_path):
            data = crime_window(start_times[i], start_times[i+1], zipcodes, crime_types)
            logger.info('{} crimes found for {}'.format(len(data), start_times[i]))
            g = network_creation.distance_graph(data, distance, node_type)
            save_graph(g, network_path)
        else:
            logger.info('Network exists, skipping')
        i += 1


def save_dynamic_distance_year_graph(years, area_name, distance, node_type, crime_types=None):
    """ Saves a number of networks, where each network represents a year.

        Each network is saved to disk at a location that matches the parameters
        used to construct it. This method uses the box network method to make
        network creation faster. This means each network may not contain all
        the crimes available for that period of time.

        :param years: The time frame of networks to create. Should be a list of
        integers.
        :param area_name: The name of the area to search for crimes. This
        should be a valid entry in cities.json
        :param distance: The maximum distance between two connected crimes.
        :param node_type: What each node represents. The is passed on to
        network_creation.distance_graph.
        :param crime_types: The types of crimes to include in the network.

        Examples
        --------
        >>> save_dynamic_distance_year_graph([2008, 2009], 'miami', 1.6, 'crime', ['Theft'])
    """
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
        network_path = '{}/networks/{}_{}.graphml'.format(path, 'year', date_string(start_times[i]))
        if not os.path.exists(network_path):
            limits['date'] = {'$gte': start_times[i], '$lt': start_times[i+1]}
            logger.info('Building {}'.format(limits))
            g = box_networks.distance_crime_network_by_box(distance, area_name, limits=limits)
            save_graph(g, network_path)
        else:
            logger.info('Network exists, skipping')
        i += 1


if __name__ == '__main__':
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))
    # logging.basicConfig(level=logging.DEBUG)

    todo = 'collapse'

    areas = ['baltimore', 'los_angeles', 'miami']
    distances = [0.1, 0.8, 1.6, 2.4, 3.2]
    node_types = ['crime']
    _crime_types = [None, ['Theft'], ['Burglary'], ['Assault']]

    logger.info('Starting')
    if todo == 'month':
        params = multithreading.combinations(years=[range(2007, 2011)],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=_crime_types)
        multithreading.map_kwargs(save_dynamic_distance_month_graph, params)
    elif todo == 'week':
        params = multithreading.combinations(initial=[datetime.datetime(2007, 1, 1)],
                                             final=[datetime.datetime(2011, 1, 1)],
                                             delta_name=['week'],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=_crime_types)
        logger.info('Generating {} dynamic networks'.format(len(params)))
        multithreading.map_kwargs(save_dynamic_distance_delta_graph, params)
        # map(lambda args: save_dynamic_distance_graph(**args), params)
    elif todo == 'year':
        params = multithreading.combinations(years=[range(2007, 2011)],
                                             area_name=areas,
                                             distance=distances,
                                             node_type=node_types,
                                             crime_types=_crime_types)
        map(lambda args: save_dynamic_distance_year_graph(**args), params)
    elif todo == 'collapse':
        collapse_all_to_zip()
    logger.info('Done!')
