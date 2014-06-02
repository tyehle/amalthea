__author__ = 'Tobin Yehle'

import math
import igraph
from datetime import datetime
import network_creation
from windows import crime_window

anaheim = ['92801', '92802', '92803', '92804', '92805', '92806', '92807',
           '92808', '92809', '92812', '92814', '92815', '92816', '92817',
           '92825', '92850', '92899']
long_beach = ['90801', '90802', '90803', '90804', '90805', '90806',
              '90807', '90808', '90809', '90810', '90813', '90814',
              '90815', '90822', '90831', '90832', '90833', '90834',
              '90835', '90840', '90842', '90844', '90845', '90846',
              '90847', '90848', '90853', '90888', '90899']
la = ['90001', '90002', '90003', '90004', '90005', '90006', '90007',
      '90008', '90009', '90010', '90011', '90012', '90013', '90014',
      '90015', '90016', '90017', '90018', '90019', '90020', '90021',
      '90022', '90023', '90024', '90025', '90026', '90027', '90028',
      '90029', '90030', '90031', '90032', '90033', '90034', '90035',
      '90036', '90037', '90038', '90039', '90040', '90041', '90042',
      '90043', '90044', '90045', '90046', '90047', '90048', '90049',
      '90050', '90051', '90052', '90053', '90054', '90055', '90056',
      '90057', '90058', '90059', '90060', '90061', '90062', '90063',
      '90064', '90065', '90066', '90067', '90068', '90070', '90071',
      '90072', '90073', '90074', '90075', '90076', '90077', '90078',
      '90079', '90080', '90081', '90082', '90083', '90084', '90086',
      '90087', '90088', '90089', '90091', '90093', '90094', '90095',
      '90096', '90097', '90099', '90101', '90102', '90103', '90174',
      '90185', '90189', '90291', '90292', '90293', '91040', '91041',
      '91042', '91043', '91303', '91304', '91305', '91306', '91307',
      '91308', '91342', '91343', '91344', '91345', '91346', '91347',
      '91348', '91349', '91352', '91353', '91356', '91357', '91364',
      '91365', '91366', '91367', '91401', '91402', '91403', '91404',
      '91405', '91406', '91407', '91408', '91409', '91410', '91411',
      '91412', '91413', '91414', '91415', '91416', '91417', '91418',
      '91419', '91420', '91421', '91422', '91423', '91424', '91425',
      '91426', '91427', '91428', '91429', '91430', '91431', '91432',
      '91433', '91434', '91435', '91436', '91437', '91438', '91439',
      '91440', '91441', '91442', '91443', '91444', '91445', '91446',
      '91447', '91448', '91449', '91450', '91451', '91452', '91453',
      '91454', '91455', '91456', '91457', '91458', '91459', '91460',
      '91461', '91462', '91463', '91464', '91465', '91466', '91467',
      '91468', '91469', '91470', '91471', '91472', '91473', '91474',
      '91475', '91476', '91477', '91478', '91479', '91480', '91481',
      '91482', '91483', '91484', '91485', '91486', '91487', '91488',
      '91489', '91490', '91491', '91492', '91493', '91494', '91495',
      '91496', '91497', '91498', '91499', '91601', '91602', '91603',
      '91604', '91605', '91606', '91607', '91608', '91609']

# Baltimore zip codes
balt_list = range(21201, 21232) + range(21233,21238) + range(21239,21242) + [21244] +  range(21250,21253) + range(21263,21266) + [21268] + [21270] + range(21273,21276) + range(21278,21291) + [21297, 21298]
for i in range(len(balt_list)):
  balt_list[i] = str(balt_list[i])

def layout_position(g):
    g.vs['x'] = [float(x) for x in g.vs['longitude']]
    g.vs['y'] = [-float(y) for y in g.vs['latitude']]


def get_bounds(g, max_dimension=1600):
    xs = [float(x) for x in g.vs['longitude']]
    ys = [-float(y) for y in g.vs['latitude']]
    bounds = (0, 0, max(xs) - min(xs), max(ys) - min(ys))
    scale_factor = max_dimension / max(bounds)
    return [int(x*scale_factor) for x in bounds]


def save_networks():
    # grab the data we will use
    data = crime_window(zipcodes=la+anaheim+long_beach,
                        start_date=datetime(2010, 12, 1),
                        end_date=datetime(2011, 1, 1))

    distances = [.1, .8, 1.6, 2.4, 3.2]
    distances.reverse()
    for d in distances:
        print('\ndistance = {0}\n'.format(d))
        g = network_creation.distance_zip_graph(data, d)
        g.write_graphml('output/la_zip_distance_{0}.graphml'.format(d))


def analyze_graphs():
    """ Run analysis on saved networks. """
    path = 'output/{0}_{1}_distance_{2}.graphml'
    cities = ['la', 'baltimore']
    types = ['crime', 'zip']
    distances = [.1, .8, 1.6, 2.4, 3.2]

    output_file = open('output/graph_properties.csv', 'w')

    output_file.write('City,Type,Distance,Nodes,Edges,Transitivity,Degree Assortativity,Diameter,Giant Component\n')
    output = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'

    # run the analysis on all the networks we have
    for c in cities:
        for t in types:
            for d in distances:
                f = path.format(c, t, d)
                g = igraph.Graph.Read(f)
                stats = output.format(c,
                                      t,
                                      d,
                                      g.vcount(),
                                      g.ecount(),
                                      g.transitivity_undirected(),
                                      g.assortativity_degree(),
                                      g.diameter(),
                                      g.components().giant().vcount())
                print(stats)
                output_file.write(stats)


if __name__ == '__main__':
    analyze_graphs()
