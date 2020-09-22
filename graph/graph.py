import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from distance_weighted_recommender.config.parser import Parser

class Graph(object):

	def __init__(self, transactions):

		config = Parser("config.toml").load()

		self._max_distance = \
			config["graph"]["max_distance"]

		self._transactions = transactions

		self._create_customer_trust_matrix()


	def _create_adjacency_matrix(self):

		self._adjacency_matrix = np.zeros(
			(self._transactions.shape[0], self._transactions.shape[0]),
			dtype=np.bool,
		)

		list_of_neighbour_customers = [ np.nonzero(t)[0] for t in self._transactions.T ]

		for neighbour_customers in list_of_neighbour_customers:
			for i in range(neighbour_customers.shape[0]):
				self._adjacency_matrix[neighbour_customers[i], neighbour_customers[i+1:]] = \
					self._adjacency_matrix[neighbour_customers[i+1:], neighbour_customers[i]] = True


	def _create_distance_matrix(self):

		self._create_adjacency_matrix()

		self._adjacency_matrix = csr_matrix(self._adjacency_matrix)
		self._distance_matrix = \
			dijkstra(csgraph=self._adjacency_matrix, 
					directed=False, 
					return_predecessors=False, 
					unweighted=True,
					limit=self._max_distance)

		self._distance_matrix[~np.isfinite(self._distance_matrix)] = 0

	"""
	def _create_customer_filterer_matrix(self):

		self._customer_filterer_matrix = \
			(self._distance_matrix <= self._max_distance) & (self._distance_matrix > 0)
	"""

	def _create_customer_trust_matrix(self):

		self._create_distance_matrix()

		"""
		self._create_customer_filterer_matrix()
		"""

		self._customer_trust_matrix = \
			np.reciprocal(self._distance_matrix, out=np.zeros_like(self._distance_matrix), where=self._distance_matrix!=0)

		"""
		self._customer_trust_matrix *= self._customer_filterer_matrix
		"""