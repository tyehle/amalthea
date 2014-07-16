__author__ = 'Tobin Yehle'

import logging
import compare_borders
import matplotlib.pyplot as plt
import igraph

logger = logging.getLogger(__name__)


def correlation_progression(g, iterations, conserve_weight=False):
    if iterations == -1:
        iterations = g.ecount()

    base = g.copy()

    progression = []
    diffs = []
    for i in range(iterations):
        # logger.debug('{:.3g} %'.format(i * 100.0 / iterations))
        if conserve_weight:
            compare_borders.randomize_borders_conserve_weight(g, iterations=1)
            diffs.append(0)
        else:
            diff = compare_borders.randomize_borders(g, iterations=1)
            if len(diff) == 1:
                diffs.append(diff[0])
            else:
                diffs.append(0)
        progression.append(compare_borders.get_absolute_cross_correlation(base, g))

    return progression, diffs


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    _path = 'data/miami/all/distance/1.6/zip'
    # _filename = 'week_2009-10-05'
    # _filename = 'week_2010-12-20'
    _filename = 'month_2010-12-01'

    _network = igraph.Graph.Read('{}/borders/zip/label_propagation/{}_1000.graphml'.format(_path, _filename))
    _iterations = int(_network.ecount() * 16)

    logger.info('Finding correlation')
    _conserve = _network.copy()
    _change = _network.copy()
    _data_change, _diffs = correlation_progression(_change, _iterations, conserve_weight=False)
    _data_conserve, _ = correlation_progression(_conserve, _iterations, conserve_weight=True)

    logger.info('Plotting')
    prog = plt.subplot(221)
    prog.plot(_data_change)
    prog = plt.subplot(222, sharey=prog)
    prog.plot(_data_conserve, 'r')

    dist = plt.subplot(223)
    dist.hist(_change.es['weight'], bins=40)
    dist = plt.subplot(224, sharey=dist)
    dist.hist(_conserve.es['weight'], bins=40, color='r')

    plt.show()

    import numpy
    plt.plot(numpy.cumsum(_diffs))
    plt.show()
