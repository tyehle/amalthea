import logging.config
import math
import random
import save_networks
import datetime
import multithreading
import igraph
import itertools
import csv
import os.path
import json

logger = logging.getLogger(__name__)


def randomize_borders_conserve_weight(b, iterations=-1, choose_prob=0.5):
    """ Randomizes the borders in a border network, conserving the overall
        weight of all edges in the network.

        This changes the given network into a random border network.

        :param b: The border network to randomize.
        :param iterations: The number of iterations to run the randomization
        process. If this is -1 (the default) then the process will run for the
        number of edges in the network.
        :param choose_prob: The probability of choosing a border to subtract
        weight from.

        Examples
        --------
        >>> import igraph
        >>> r = igraph.Graph.Read('borders.graphml')
        >>> b = igraph.Graph.Read('borders.graphml')
        >>> randomize_borders_conserve_weight(r)
        >>> cc = get_absolute_cross_correlation(b, r)
    """
    if iterations == -1:
        iterations = b.ecount() * 16

    for i in range(iterations):
        # choose a random vertex
        v = b.vs[random.randint(0, b.vcount() - 1)]

        # randomly split the incident edges into two groups
        add = sub = []
        for e in b.es:
            if v['zipcode'] != b.vs[e.source]['zipcode'] and \
               v['zipcode'] != b.vs[e.target]['zipcode']:
                # this edge does not interest us
                continue
            if random.random() < choose_prob:
                sub.append(e)
            else:
                add.append(e)

        if len(add) == 0:
            # no weight can be redistributed
            continue

        to_remove = [random.random() * e['weight'] for e in sub]
        to_add = sum(to_remove) / len(add)

        # redistribute weight
        for e, amount in zip(sub, to_remove):
            if e['weight'] != 0:
                e['weight'] -= amount
        for e in add:
            e['weight'] += to_add


def randomize_borders(b, iterations=-1):
    """ Randomizes the borders in a border network.

        This changes the given network into a random border network.

        :param b: The border network to randomize.
        :param iterations: The number of iterations to run the randomization
        process. If this is -1 (the default) then the process will run for the
        number of edges in the network.

        Examples
        --------
        >>> import igraph
        >>> r = igraph.Graph.Read('borders.graphml')
        >>> b = igraph.Graph.Read('borders.graphml')
        >>> randomize_borders(r)
        >>> cc = get_absolute_cross_correlation(b, r)
    """
    if iterations == -1:
        iterations = b.ecount() * 16

    diffs = []
    for i in range(iterations):
        # choose a random vertex
        v = b.vs[random.randint(0, b.vcount() - 1)]

        # randomly split the incident edges into two groups
        es = []
        for e in b.es:
            if v['zipcode'] == b.vs[e.source]['zipcode'] or \
               v['zipcode'] == b.vs[e.target]['zipcode']:
                # this edge is incident on the vertex
                es.append(e)

        if len(es) <= 1:
            # no weight can be redistributed
            continue

        sub = random.sample(es, random.randint(1, len(es) - 1))
        add = [e for e in es if e not in sub]

        remove_possible = [e['weight'] for e in sub if e['weight'] != 0]
        if len(remove_possible) != 0:
            to_add = random.random() * min(remove_possible)
        else:
            to_add = 0

        # redistribute weight
        for e in sub:
            if e['weight'] != 0:
                e['weight'] -= to_add
        for e in add:
            e['weight'] += to_add

        diffs.append(to_add * (len(add) - len([e for e in sub if e['weight'] != 0])))

    return diffs


def get_absolute_cross_correlation(a, b):
    """ Finds the absolute cross correlation between two networks with the same
        structure.

        The two given networks must have nodes with a zipcode attribute. The
        zip codes of the source and targets of all edges in a must match those
        in b.

        :return: The absolute cross correlation between a and b.

        Examples
        --------
        >>> import igraph
        >>> a = igraph.Graph.Read('crimes.graphml')
        >>> b = igraph.Graph.Read('census.graphml')
        >>> cc = get_absolute_cross_correlation(a, b)
    """
    # each node must have a zipcode attribute
    # the two networks must have the same structure
    if a.ecount() != b.ecount():
        raise AssertionError('a and b do not have the same number of edges!')
    prod = aprod = bprod = 0.0
    # find a sortable list of edges for both networks ((zip, zip), w)
    get_sortable_edge = lambda g, e: (tuple(sorted([g.vs[e.source]['zipcode'],
                                                    g.vs[e.target]['zipcode']])),
                                      e['weight'])

    a_es = [get_sortable_edge(a, ae) for ae in a.es]
    b_es = [get_sortable_edge(b, be) for be in b.es]
    a_es.sort()
    b_es.sort()
    for i in range(len(a_es)):
        if a_es[i][0] != b_es[i][0]:
            logger.error('Non-matching Edge sets:\n\t{}\n\t{}'.format(a_es, b_es))
            raise AssertionError('Edges do not match!')
        prod += a_es[i][1] * b_es[i][1]
        aprod += a_es[i][1]**2
        bprod += b_es[i][1]**2
    return prod / math.sqrt(aprod * bprod)


def get_z_score(base, randoms, target):
    exists = [v['zipcode'] for v in target.vs]

    def filtered(g):
        f = g.copy()
        f.delete_vertices(f.vs.select(lambda v: v['zipcode'] not in exists))
        return f

    base = filtered(base)

    rand_correlations = [get_absolute_cross_correlation(base, filtered(r)) for r in randoms]

    mean = sum(rand_correlations) / float(len(rand_correlations))
    std_dev = math.sqrt(sum((c-mean)**2 for c in rand_correlations) / float(len(rand_correlations)))

    return (get_absolute_cross_correlation(target, base) - mean) / std_dev


def get_z_scores(area, clustering, level, crime_types, distances, node_types, filenames,
                 algorithms, iterations_list, randomizations=10):
    def targets():
        """ Generator function to yield each target network one at a time. """
        for params in itertools.product(crime_types,
                                        distances,
                                        node_types,
                                        filenames,
                                        algorithms,
                                        iterations_list):
            (crime_type, distance, node_type, filename, algorithm, iterations) = params
            path = 'data/{}/{}/distance/{}/{}'.format(area, crime_type, distance, node_type)
            border_path = '{}/borders/zip/{}/{}_{}.graphml'.format(path, algorithm, filename, iterations)
            if os.path.exists(border_path):
                yield igraph.Graph.Read(border_path), params
            else:
                logger.warning('Borders Not Found! {}'.format(border_path))

    base_path = 'data/{0}/census/hierarchical/{1}/{1}_{2}.graphml'.format(area, clustering, level)
    base = igraph.Graph.Read(base_path)

    def get_random_copy(b):
        r = b.copy()
        randomize_borders(r)
        return r

    logger.info('Generating Randomized Networks')
    randoms = [get_random_copy(base) for _ in range(randomizations)]

    logger.info('Comparing Base to Targets')
    return [params + (get_z_score(base, randoms, g),) for (g, params) in targets()]


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.config.dictConfig(json.load(open('logging_config.json', 'r')))

    week_files = save_networks.week_files(datetime.datetime(2007, 1, 1), datetime.datetime(2011, 1, 1))
    month_files = save_networks.month_files(range(2007, 2011))
    year_files = save_networks.year_files(range(2007, 2011))

    logger.debug(year_files)

    _cities = ['baltimore', 'los_angeles', 'miami']
    _clusterings = ['average', 'single', 'complete']
    _levels = [50000, 25000]

    params = multithreading.combinations(area=_cities,
                                         clustering=_clusterings,
                                         level=_levels,
                                         crime_types=[['all', 'assault', 'burglary', 'theft']],
                                         distances=[[3.2, 2.4, 1.6, 0.8, 0.1]],
                                         node_types=[['zip']],
                                         filenames=[year_files],
                                         algorithms=[['label_propagation']],
                                         iterations_list=[[1000]])

    # filter out bad combinations
    params = filter(lambda args: args['clustering'] == 'average' and args['level'] == 50000 or
                                 args['clustering'] == 'single' and args['level'] == 25000 or
                                 args['clustering'] == 'complete' and args['level'] == 25000,
                    params)

    logger.info('{} Base Networks Found'.format(len(params)))

    score_lists = multithreading.map_kwargs(get_z_scores, params)
    logger.info('Combining Results')
    data = [('area', 'clustering', 'level', 'crime_type', 'distance', 'node_type',
             'filename', 'algorithm', 'iterations', 'zscore')]  # fill with the header row
    logger.debug(params)
    logger.debug(score_lists)
    for args, results in zip(params, score_lists):
        for score in results:
            data.append((args['area'], args['clustering'], args['level']) + score)

    logger.info('Writing Data File')
    # write csv
    with open('border_comparason.csv', 'a') as output:
        writer = csv.writer(output, delimiter=',')
        for row in data:
            writer.writerow(row)