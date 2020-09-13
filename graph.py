import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from tacorec.config.config_parser import ConfigParser

class Graph(object):

	def __init__(self, transactions):

		config = ConfigParser("tacorec.toml").load()

		self._max_distance = \
			config["graph"]["max_distance"]

		self._transactions = transactions

		self._create_adjacency_matrix()
		self._create_distance_matrix()
		self._create_customer_filterer_matrix()
		self._create_customer_trust_matrix()

	def _create_adjacency_matrix(self):

		self._adjacency_matrix = np.zeros(
			(self._transactions.shape[0], self._transactions.shape[0]),
			dtype=np.bool,
		)

		for i in range(self._transactions.shape[0]):
			for j in range(i + 1, self._transactions.shape[0]):
				if np.sum(np.logical_and(self._transactions[i], self._transactions[j])) != 0:
					self._adjacency_matrix[i][j] = self._adjacency_matrix[j][i] = True


	def _create_distance_matrix(self):

		self._adjacency_matrix = csr_matrix(self._adjacency_matrix)
		self._distance_matrix = \
			dijkstra(csgraph=self._adjacency_matrix, directed=False, return_predecessors=False)


	def _create_customer_filterer_matrix(self):

		self._customer_filterer_matrix = self._distance_matrix <= self._max_distance


	def _create_customer_trust_matrix(self):

		self._customer_trust_matrix = \
			np.reciprocal(self._distance_matrix, out=np.zeros_like(self._distance_matrix), where=self._distance_matrix!=0)

		self._customer_trust_matrix *= self._customer_filterer_matrix