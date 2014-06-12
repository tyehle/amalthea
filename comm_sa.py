import igraph
from math import exp
from random import randint, random, sample
from pymongo import MongoClient
import shapely.geometry

client = MongoClient('163.118.78.22', 27017)
db = client['crimes_test']
geometry = db.geometry

def community_SA(g, T0 = 2.5 *10**-4, c = 0.75, f = 0.5):
	""" Partitions the graph using the SA community detection algorithm proposed
		in Guimera and Amaral's publication.

		Assumptions made when implementing the algorithm include randomly 
		selecting the node n for which to locally modify and using a 50% 
		to determine whether a global split or merge is proposed. The split
		algorthim follows the detection algorith exactly. For each T, f * S**2
		local changes are made.

		Parameters
		----------
		g: igraph.Graph
			The graph of interest.
		T0: float
			The intial temperature. The default is 2.5 * 10**-4, as proposed in 
			Brockman's supplmental materials.
		c: float
			The cooling factor. The default is c = 0.75, as proposed in Brockman's
			supplemental materials.
		f: float
			The proportional of changes made. The default is 0, as proposed in
			Brockman's paper.

		Returns
		-------
		igraph.VertexClustering
			Returns a clustering of the vertex set of the graph.

		References
        ----------
        R. Guimera, L. Amaral

		Examples
		--------
		>>> part_SA = community_SA(g, f = 0.65)
	"""
	T = float(T0)
	S = g.vcount()
	# Initialize P such that each N node is in its own module
	P = [node.index for node in g.vs]
	accept = False
	list_steps = []
	while(not accept):
		print 'temp: {}    modularity: {}'.format(T, mod_calc(P, g))
		# Propose fS**2 individual node movements
		for i in range(int(f * S**2)):
			Pnew = local_update(list(P) , g)
			# Accept new partition according to equation 2 from the publication
			if accept_update(g, Pnew, P, T):
				P = list(Pnew)
		print 'Done with local...'
		# Propose fS collective movements 
		for i in range(int(f * S)):
			# Half of the time, merge modules
			if randint(1,2) == 1:
				Pnew = merge_update(list(P))
				# Accept new partition according to equation 2 from the publication
				if accept_update(g, Pnew, P, T):
			 		# Test merging:
			 		P = list(Pnew)
			 		print 'Merge accepted!'
			# Half of the time, split modules
			else:
				# Split module using simplified SA community detection algorithm
				Pnew = split_update(igraph.VertexClustering(g, P), sample(list(P), 1)[0], T0, T, S, c, f, g)
				# Accept new partition according to equation 2 from the publication
				if accept_update(g, Pnew, P, T):
					P = list(Pnew) 
					print 'Split accepted!'
		# Append current modularity
		list_steps.append(mod_calc(P, g))
		# Check if modularity has improved within three last temp steps
		if len(list_steps) > 3:
			# Maintain length of 3
			list_steps.remove(list_steps[0])
			if (abs(list_steps[0] - list_steps[1]) + abs(list_steps[0] - list_steps[2])) < 2 * 10**-3: 
				# If M has seen no improvement, accept the partition and exit the while loop
				accept = True
		# Cool T
		T *= c
		# Look at plot each time
		igraph.plot(igraph.VertexClustering(g, P), mark_groups = True)
	return igraph.VertexClustering(g, P)

def accept_update(g, Pnew, P, T):
	""" Returns boolean as to accept or reject partition update.

		Calculates the probability of a proposed partition being accepted. The 
		probability formula is as follows: if the cost after the update is <= 
		the cost before the update, the parition is accepted with probability 
		one, otherwise the parition is accepted with probability e**(-(Cf - Ci)/T). 
		Note that C = -M.

		Parameters
        ----------
        g: igraph.Graph
        	Graph of interest.
        Pnew: list
        	Proposed modification to the partition of g.
        P: list 
        	Current partition of g.
        T: float
        	The current temperature.

        Returns
        -------
        boolean
        	Whether or not Pnew should be accepted as the new partition of g.

        References
        ----------
        R. Guimera, L. Amaral
   	"""
	Mi = mod_calc(P, g)
	Mf = mod_calc(Pnew, g)
	# Probability of accpeting new partition is one if cost has decreased or stayed the same
	if -Mf <= -Mi:
		return True
    # Calculate probability if cost has increased
	p = exp(T**-1 * (Mf - Mi))
    # Accept partition with probability p
	if random() >= (1-p):
		return True
	else:
		return False

def local_update(mem_list, g):
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
	n = sample([node.index for node in g.vs], 1)[0]
	# Assign n to random module m
	mem_list[n] = sample(mem_list, 1)[0]
	return mem_list

def merge_update(mem_list):
	""" Peforms a merge update given a partition.

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


def split_update(P, m, T0, Tcurr, S, c, f, g):
	""" Peforms a split update given a partition.

		This method uses a simplified version of the overall SA community 
		detection algorithm. Until the original time has been cooled to the
		current time of the overall algorithm, a sub graph of the module of 
		interest is modified using local changes. The sub graph is initialized
		with two randomly assigned partitions.

		Parameters
		----------
		P: igraph.VertexClustering
			Current partition of g.
		m: int
			Module of interest.
		T0: float
			Original time of overall community detection algorithm.
		Tcurr: float
			Current time of overall community detection algorithm.
		S: int
			Number of nodes in the graph of interest.
		c: float
			The cooling factor. The default is c = 0.75, as proposed in Brockman's
			supplemental materials.
		f: float
			The proportional of changes made. The default is 0.5 as proposed in
			Brockman's paper.

		Returns
		-------
		list
			Proposed modification to the partition of g.
	"""
	T = float(T0)
	# Obtain the subgraph of nodes of community m
	g_sub = P.subgraph(m)
	# Create P_sub such that nodes are assigned to 2 random modules
	P_sub = [randint(1,2) for i in g_sub.vs]
	while(T > Tcurr):
		# Propose fS**2 individual node movements
		for i in range(int(f * S**2)):
			Pnew_sub = list(P_sub)
			# Choose random node n
			n = sample([i.index for i in g_sub.vs], 1)[0]
			# Assign n to random module m
			mod = sample(Pnew_sub, 1)[0]
			Pnew_sub[n] = mod
			# Accept new partition according to equation 2 from the publication
			if accept_update(g_sub, Pnew_sub, P_sub, T):
				P_sub = list(Pnew_sub)
		# Cool T
		T *= c
	# Assign all nodes in community m to their new community 
	member_new = P.membership
	inc = max(member_new)
	# Save indices of nodes originally in community m
	m_id = [i for i in range(len(member_new)) if member_new[i] == m]
	# Traverse list of indices, input incremented new module number/label 
	for i, loc in enumerate(m_id):
		member_new[loc] = inc + P_sub[i] 
	# Return new membership list
	if [P_sub[0] for i in P_sub] == P_sub:
		print 'Split returned the same partition...'
		if 0 in [g.degree(i) for i in m_id]:
			print '...and there were isolated nodes in the partition'
			print '...{} components'.format(len(g_sub.components()))
	return member_new

def mod_calc(P_list, g):
	""" Calculates the modularity of a partition of g.

		Uses the constraints implied in the publication as well as constraints
		necessary to partition a graph that is disconnnected. The given 
		constraints include: returning a modularity of one to paritions of a 
		single modules containing a single isolated node, returning a modularity 
		of zero to partitions of multiple isolated modules, incrementing the 
		modularity by 1/# of modules for a module that contains a single isolated
		node. Otherwise, the modularity is calculated as proposed in equation 1.

		Parameters
		----------
		P: list
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
	P = igraph.VertexClustering(g, P_list)
	L = float(g.ecount())
	# Return modularity of 1 for module containing single isolated node
	if L == 0 and g.vcount == 1:
		return 1
	# Return modularity of 0 for module containing multiple isolated nodes
	if L == 0 and g.vcount > 1:
		return 0
	# Calculate modularity
	M = 0
	for i, mod in enumerate(P):
		# Skip if empty module
		if len(mod) == 0:
				continue
		# Create subgraph containing module of interest
		g_sub = g.subgraph(mod)
		# A module that contains a single isolated node adds 1/# of modules
		if g_sub.vcount() == 1 and g.degree(P[i][0]) == 0:
			M += len(P)**-1
			continue
		ls = float(g_sub.ecount())
		ds = float(sum(g_sub.degree()))
		# Penalty is proportional to the number of components in subgraph
		penalty = len(g_sub.components())
		M += (ls / L - ((ds / (2*L)) ** 2))/penalty
	return M 

def conn_graph(g_given):
	""" Returns a graph with a giant component equal to the whole graph.

		All geographically adjacent zip codes are given an edge of weight 0.5
		if such nodes do not already have an edge. The geographically adjacent 
		edges are found using the shapefiles from the geometry database.

		Parameters
		----------
		g_given: igraph.Graph
			The zip code graph of interest.

		Returns
		-------
		igraph.Graph 
			The same given graph but with edges between all geographically 
			adjacent zip codes.
		"""
	g = g_given
	zip_list = []
	# Traverse all zip code's borders
	for z in range(len(g.vs['zipcode'])):
		# Add shapefile to list from geometry database
		zip_list.append(shapely.geometry.asShape(geometry.find_one({'zip': g.vs['zipcode'][z]})['geometry']))
		print 'at {} out of {}'.format(z, len(g.vs['zipcode']) - 1)
		# Compare borders to all other borders
		for i in range(len(zip_list) - 1):
			# Test for intersection between zip code borders
			if zip_list[z].intersects(zip_list[i]):
				n = g.vs.select(zipcode_eq = g.vs['zipcode'][z])[0].index
				n_other = g.vs.select(zipcode_eq = g.vs['zipcode'][i])[0].index
				# Test for existing edge between nodes
				if len(g.es.select(_within=[n,n_other])) == 0:
					# Create edge with weight 0.001
					print 'adding edge between node {} and node {}'.format(n, n_other)
					g.add_edge(n, n_other, weight = 0.5)
					# g.es[g.ecount()-1]['weight'] = 0.5
	return g
					
if __name__ == '__main__':
	g = igraph.Graph.Read_GraphML('/Users/swhite/Documents/Amalthea/data/baltimore/distance/2.4/zip/networks/17dec2010.graphml')
	g.vs['x'] = [float(x) for x in g.vs['longitude']]
	g.vs['y'] = [-float(y) for y in g.vs['latitude']]
	igraph.plot(g)
	part = community_SA(g)
