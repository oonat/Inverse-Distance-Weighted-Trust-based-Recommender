import numpy as np

from tacorec.database import neo4j_interface
from tacorec.config.config_parser import ConfigParser
from tacorec.network_filtering.network_filterer import NetworkFilterer

from tacorec.distance_oriented_recommender.tester.graph import Graph
from scipy.sparse import csr_matrix

import random


class TrustBasedFilterer(object):

	def __init__(self, sales):

		self._interface = neo4j_interface.ApplicationInterface()

		""" get the parameters from the config file """
		config = ConfigParser("tacorec.toml").load()

		self._number_of_recommendations = \
			config["trust_based_recommendation"]["recommendations_per_user"]

		filtering_threshold = 3 
		"""config["network_filtering"]["filtering_threshold"]"""


		self._sales = np.array(sales, dtype=np.uint32)
		network_filterer = NetworkFilterer(self._sales, filtering_threshold)
		self._unique_customers, self._unique_products = \
			network_filterer.filter_network()
		self._sales = network_filterer.encode_to_consecutive_ids()


		self._create_customers_versus_products_table()


	def _create_customers_versus_products_table(self):

		self._customers_versus_products_table = np.zeros(
			(self._unique_customers.shape[0], self._unique_products.shape[0]),
			dtype=np.bool,
		)
		self._customers_versus_products_table[
			self._sales[:, 0],
			self._sales[:, 1],
		] = True


	def _precalculate_magnitudes(self):

		self._precalculated_magnitudes = np.empty(
			self._unique_customers.shape,
			dtype=np.float32,
		)

		for i, row in enumerate(self._customers_versus_products_table):
			self._precalculated_magnitudes[i] = np.sqrt(np.sum(row))


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
				self._unique_customers.shape[0],
				self._unique_customers.shape[0],
			),
			dtype=np.float32,
		)

		for i in range(self._unique_customers.shape[0]):
			for j in range(i + 1, self._unique_customers.shape[0]):
				similarity_coefficient = self._calculate_similarity_coefficient(i, j)

				self._similarity_matrix[i][j] = self._similarity_matrix[j][i] = similarity_coefficient


		self._similarity_matrix = \
			csr_matrix(self._similarity_matrix).multiply(self._graph._customer_filterer_matrix)


	def _create_weight_matrix(self):

		self._create_similarity_matrix()

		numerator = 2 * self._graph._customer_trust_matrix * self._similarity_matrix
		denominator = self._graph._customer_trust_matrix + self._similarity_matrix

		self._weight_matrix = \
			np.array(numerator / denominator)

		self._weight_matrix[np.isnan(self._weight_matrix)] = 0


	def _calculate_product_coefficients(self, customer):

		coefficient_list = []

		for product in range(self._unique_products.shape[0]):
			if self._customers_versus_products_table[customer][product]:
				continue

			product_array = np.delete(self._customers_versus_products_table[:,product], customer, 0)
			weight_array = np.delete(self._weight_matrix[customer], customer, 0)
			numerator = np.sum(product_array * weight_array)
			denominator = np.sum(weight_array)
			recommendation_coefficient = 0 if denominator == 0 else float(numerator)/denominator 
			coefficient_list.append((self._unique_products[product], recommendation_coefficient))

		return coefficient_list



	def decide_deleted_product(self, customer):

		customer_row = self._customers_versus_products_table[customer,:]
		bought_products = customer_row.nonzero()[0]

		return random.choice(bought_products)


	def make_recommendation_to_customer(self, customer):

		deleted_product = self.decide_deleted_product(customer)
		deleted_product_id = self._unique_products[deleted_product]

		self._customers_versus_products_table[customer][deleted_product] = 0

		self._graph = Graph(self._customers_versus_products_table)
		self._create_weight_matrix()

		products_with_coefficients = self._calculate_product_coefficients(customer)
		products_with_coefficients.sort(key = lambda x: x[1], reverse=True)  

		product_list = []
		coefficient_list = []

		for product, coefficient in products_with_coefficients[:self._number_of_recommendations]:
			product_list.append(product)
			coefficient_list.append(coefficient)

		self._customers_versus_products_table[customer][deleted_product] = 1

		return (self._unique_customers[customer], deleted_product_id, product_list, coefficient_list)


	def make_recommendations(self):

		for customer in range(self._unique_customers.shape[0]-200):
			yield self.make_recommendation_to_customer(customer)
