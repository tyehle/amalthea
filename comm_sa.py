import igraph
from math import exp
from random import randint, random, sample

def community_sa(g, t0 = 2.5 *10**-4, C = 0.75, f = 0.5):
    """ Partitions the graph using the SA community detection algorithm proposed in Guimera and Amaral's publication.

        Assumptions made when implementing the algorithm include randomly selecting the node n for which to locally modify and using a 50% to determine whether a global split or merge is proposed. The splitalgorthim follows the detection algorith exactly. For each T, f * S**2 local changes are made.

        Parameters
        ----------
        g: igraph.Graph
            The graph of interest.
        T0: float
            The intial temperature. The default is 2.5 * 10**-4, as proposed in Brockman's supplmental materials.
        c: float
            The cooling factor. The default is c = 0.75, as proposed in Brockman's supplemental materials.
        f: float
            The proportional of changes made. The default is 0, as proposed in Brockman's paper.

        Returns
        -------
        igraph.VertexClustering
            Returns a clustering of the vertex set of the graph.

        References
        ----------
        R. Guimera, L. Amaral

        Examples
        --------
        >>> part_SA = community_sa(g, f = 0.65)
    """
    t = float(t0)
    S = g.vcount()
    # Initialize p such that each N node is in its own module
    p = range(S)
    accept = False
    list_steps = []
    while(not accept):
        print 'temp: {}    modularity: {}'.format(t, mod_calc(p, g))
        print p
        # Propose fS**2 individual node movements
        for i in range(int(f * S**2)):
            pnew = _local_update(list(p))
            # Accept new partition according to equation 2 from the publication
            if _accept_update(g, pnew, p, t):
                p = list(pnew)
            if i % 100 == 0:
                print '{} of {} local updates complete'.format(i, int(f * S**2))
        # Propose fS collective movements 
        # Change probability of merge given previous proposal rejections
        merge_prob = 2
        for i in range(int(f * S)):
            # Half of the time, merge modules
            if i % 100 == 0:
                print '{} of {} local updates complete'.format(i, int(f * S))
            if randint(1, int(merge_prob)) != 1:
                pnew = _merge_update(list(p))
                # Accept new partition according to equation 2 from the publication
                if _accept_update(g, pnew, p, t):
                    # test merging:
                    p = list(pnew)
            # Half of the time, split modules
            else:   
                # Split module using simplified SA community detection algorithm
                pnew = _split_update(igraph.VertexClustering(g, p), sample(list(p), 1)[0], t0, t, S, C, f, g)
                # Accept new partition according to equation 2 from the publication
                if _accept_update(g, pnew, p, t):
                    p = list(pnew) 
                else: 
                    # For every fifth rejection 
                    merge_prob += 0.2
        # Append current modularity
        list_steps.append(mod_calc(p, g))
        # Check if modularity has improved within three last temp steps
        if len(list_steps) > 3:
            # Maintain length of 3
            list_steps.remove(list_steps[0])
            if (abs(list_steps[0] - list_steps[1]) + abs(list_steps[0] - list_steps[2])) < 2 * 10**-3: 
                # If M has seen no improvement, accept the partition and exit the while loop
                accept = True
        # Cool t
        t *= C
        # Look at plot each time
        igraph.plot(igraph.VertexClustering(g, p), mark_groups = True)
    return igraph.VertexClustering(g, p)

def _accept_update(g, pnew, p, t):
    """ Returns boolean as to accept or reject partition update.

        Calculates the probability of a proposed partition being accepted. the probability formula is as follows: if the cost after the update is <= the cost before the update, the parition is accepted with probability one, otherwise the parition is accepted with probability e**(-(Cf - Ci)/t). Note that C = -M.

        Parameters
        ----------
        g: igraph.Graph
            Graph of interest.
        pnew: list
            proposed modification to the partition of g.
        p: list 
            Current partition of g.
        t: float
            the current temperature.

        Returns
        -------
        boolean
            Whether or not pnew should be accepted as the new partition of g.

        References
        ----------
        R. Guimera, L. Amaral
    """
    mi = mod_calc(p, g)
    mf = mod_calc(pnew, g)
    # Calculate probability if cost has increased
    try:
        p = exp(t**-1 * (mf - mi))
    except OverflowError:
        print 'OverflowError'
        return False
    # Probability of accpeting new partition is one if cost <= or by probability p
    if -mf <= -mi or random() >= (1 - p):
        return True
    return False

def _local_update(mem_list):
    """ Performs a local update given a partition.

        Parameters
        ----------
        mem_list: list
            Membership list of current partition.
        g: igraph.Graph
            Graph of interest.

        Returns
        -------
        mem_list: list
            Update membership list.

        References
        ----------
        R. Guimera, L. Amaral
        """
    # Choose random node n
    n = randint(0, len(mem_list) - 1)
    # Assign n to random module m
    mem_list[n] = sample(mem_list, 1)[0]
    return mem_list

def _merge_update(mem_list):
    """ peforms a merge update given a partition.

        Parameters
        ----------
        mem_list: list
            Membership list of current partition.

        Returns
        -------
        mem_list: list
            Update membership list.

        References
        ----------
        R. Guimera, L. Amaral
        """
    # Ensure their exist more than one module
    if len(set(mem_list)) == 1:
        return mem_list
    # Select two random modules
    m = sample(set(mem_list), 2)
    # Assign all nodes in module m[0] to module m[1]
    return [m[1] if n is m[0] else n for n in mem_list]


def _split_update(p, m, t0, tcurr, S, C, f, g):
    """ Peforms a split update given a partition.

        This method uses a simplified version of the overall SA community detection algorithm. Until the original time has been cooled to thecurrent time of the overall algorithm, a sub graph of the module of interest is modified using local changes. the sub graph is initialized with two randomly assigned partitions.

        Parameters
        ----------
        p: igraph.VertexClustering
            Current partition of g.
        m: int
            Module of interest.
        t0: float
            Original time of overall community detection algorithm.
        tcurr: float
            Current time of overall community detection algorithm.
        S: int
            Number of nodes in the graph of interest.
        c: float
            The cooling factor. the default is c = 0.75, as proposed in Brockman's supplemental materials.
        f: float
            The proportional of changes made. the default is 0.5 as proposed in Brockman's paper.

        Returns
        -------
        list
            Proposed modification to the partition of g.
    """
    t = float(t0)
    # Obtain the subgraph of nodes of community m
    g_sub = p.subgraph(m)
    # Create p_sub such that nodes are assigned to 2 random modules
    p_sub = [randint(1,2) for i in g_sub.vs]
    while(t > tcurr):
        # Propose fS**2 individual node movements
        for i in range(int(f * S**2)):
            pnew_sub = _local_update(list(p_sub))
            # Accept new partition according to equation 2 from the publication
            if _accept_update(g_sub, pnew_sub, p_sub, t):
                p_sub = list(pnew_sub)
        # Cool t
        t *= C
    # Assign all nodes in community m to their new community 
    mem = p.membership
    inc = max(mem)
    # Save indices of nodes originally in community m
    m_id = [i for i in range(len(mem)) if mem[i] == m]
    # traverse list of indices, input incremented new module number/label 
    for i, loc in enumerate(m_id):
        mem[loc] = inc + p_sub[i] 
    # testing code (remember to remove g from def)
    if [p_sub[0] for i in p_sub] == p_sub and g_sub.vcount() == 1:
        print 'Split returned the same partition...'
        if 0 in [g.degree(i) for i in m_id]:
            print '...the graph of interest had {} components and isolated node(s)'.format(len(g_sub.components()))
    # Return new membership list
    return mem

def mod_calc(p_list, g):
    """ Calculates the modularity of a partition of g.

        Uses the constraints implied in the publication as well as constraints necessary to partition a graph that is disconnnected. the given constraints include: returning a modularity of one to paritions of a single modules containing a single isolated node, returning a modularity of zero to partitions of multiple isolated modules, incrementing the modularity by 1/# of modules for a module that contains a single isolatednode. Otherwise, the modularity is calculated as proposed in equation 1.

        Parameters
        ----------
        p: list
            Membership list of interest.
        g: igraph.Graph
            Graph of interest.

        Returns
        -------
        M: float
            Modularity of partition. The value will be between 0 and 1.

        References
        ----------
        R. Guimera, L. Amaral
    """
    p = igraph.VertexClustering(g, p_list)
    L = float(sum(g.es['weight']))
    # Return modularity of 1 for module containing single isolated node
    if L == 0 and g.vcount == 1:
        return 1
    # Return modularity of 0 for module containing multiple isolated nodes
    if L == 0 and g.vcount > 1:
        return 0
    # Calculate modularity
    m = 0
    for i, mod in enumerate(p):
        # Skip if empty module
        if len(mod) == 0:
            continue
        # A module that contains a single isolated node adds 1/# of modules
        if len(mod) == 1 and g.degree(mod[0]) == 0:
            m += len(p)**-1
            continue
        # Create subgraph containing module of interest
        g_sub = g.subgraph(mod)
        ls = float(sum(g_sub.es['weight']))
        ds = float(sum([sum(g.es.select(_source = i)['weight']) for i in mod]))
        # Penalty applied is proportional to the number of components in subgraph
        m += ((ls / L) - (ds / L)**2)/len(g_sub.components())
    return m 

                    
if __name__ == '__main__':
    import datetime
    fs = [0.05, 0.1, 0.15]
    for f in fs:
        print f
        init = datetime.datetime.now()
        g = igraph.Graph.Read_GraphML('/Users/swhite/Documents/Amalthea/data/baltimore/distance/2.4/zip/networks/17dec2010.graphml')
        g.vs['x'] = [float(x) for x in g.vs['longitude']]
        g.vs['y'] = [-float(y) for y in g.vs['latitude']]
        igraph.plot(g)
        part = community_sa(g, f = f)
        t = (datetime.datetime.now() - init).total_seconds()
        print 'number of partitions: {}   modularity: {}   mod_cal: {}   time: {} seconds'.format(len(part), part.modularity, mod_calc(part.membership, g), t)
