import contextlib   # closing()

from tacorec.database import neo4j_interface
from tacorec.config.config_parser import ConfigParser
from tacorec.distance_oriented_recommender.tester.trust_based_filterer import TrustBasedFilterer


class TrustBasedRecommender(object):

	def __init__(self):

		self._interface = neo4j_interface.ApplicationInterface()

		self._trust_based_filterer = TrustBasedFilterer(
			list(self._get_customer_product_pairs())
			)

	
	
	def _get_customer_product_pairs(self):

		return self._interface.get_pairs(
			node_labels=("Customer", "Product"),
			path=[
				("GIVEN_BY", "left"),
				"Order",
				("OF", "right"),
			],
			properties=(
				["id"],
				["id"],
			),
		)


	def calculate_accuracy(self):

		recommendations = self._trust_based_filterer.make_recommendations()

		total = 0
		number_of_customers = 0

		i = 1
		for customer_id, deleted_product_id, products, recommendation_coefficients in recommendations:
			print(i," - ", 282)
			print(recommendation_coefficients)
			successful = 1.0/(products.index(deleted_product_id)+1) if deleted_product_id in products else 0
			total += successful
			number_of_customers += 1
			i+=1

		print("Accuracy :", total/float(number_of_customers), " Customer number: ", number_of_customers, " Total: ", total)