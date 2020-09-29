import numpy as np
from config.parser import Parser
from test.graph import Graph


class TrustBasedFilterer(object):

	def __init__(self, sales, similarity_matrix):

		config = Parser("config.toml").load()

		self._number_of_recommendations = \
			config["trust_based_recommendation"]["recommendations_per_user"]

		self._weight_ratio = \
			config["trust_based_recommendation"]["weight_ratio"]

		self._sales = np.array(sales, dtype=np.uint32)
		self._customers = np.unique(self._sales[:, 0])
		self._products = np.unique(self._sales[:, 1])

		self._similarity_matrix = np.array(similarity_matrix, dtype=np.float32)

		self._create_customers_versus_products_table()
		
		self._graph = Graph(self._customers_versus_products_table)

		self._create_weight_matrix()



	def _create_customers_versus_products_table(self):

		self._customers_versus_products_table = np.zeros(
			(self._customers.shape[0], self._products.shape[0]),
			dtype=np.uint8,
		)
		self._customers_versus_products_table[
			self._sales[:, 0],
			self._sales[:, 1],
		] = self._sales[:, 2]

	"""

	def _precalculate_magnitudes(self):

		self._precalculated_magnitudes = np.empty(
			self._customers.shape,
			dtype=np.float32,
		)

		for i, row in enumerate(self._customers_versus_products_table):
			self._precalculated_magnitudes[i] = np.sqrt(np.sum(row**2))


	def _calculate_similarity_coefficient(self, customer1, customer2):

		dot_product = np.sum(
			np.logical_and(
				self._customers_versus_products_table[customer1],
				self._customers_versus_products_table[customer2],
			),
		)

		if dot_product:
			similarity_coefficient = dot_product / (
				self._precalculated_magnitudes[customer1]
				* self._precalculated_magnitudes[customer2]
			)

		else:
			similarity_coefficient = 0

		return similarity_coefficient


	def _create_similarity_matrix(self):

		self._precalculate_magnitudes()

		self._similarity_matrix = np.zeros(
			(
				self._customers.shape[0],
				self._customers.shape[0],
			),
			dtype=np.float32,
		)

		for i in range(self._customers.shape[0]):
			for j in range(i + 1, self._customers.shape[0]):
				similarity_coefficient = self._calculate_similarity_coefficient(i, j)
				self._similarity_matrix[i][j] = \
					self._similarity_matrix[j][i] = similarity_coefficient

		self._similarity_matrix[~np.isfinite(self._graph._distance_matrix)] = 0

	"""

	def _create_weight_matrix(self):

		"""
		self._create_similarity_matrix()
		"""

		self._weight_matrix = \
			self._weight_ratio*self._graph._customer_trust_matrix + (1-self._weight_ratio)*self._similarity_matrix


