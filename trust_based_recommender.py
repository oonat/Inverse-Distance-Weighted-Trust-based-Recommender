import contextlib   # closing()

from tacorec.database import neo4j_interface
from tacorec.config.config_parser import ConfigParser
from tacorec.distance_oriented_recommender.trust_based_filterer import TrustBasedFilterer


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


	def make_recommendations(self):
		with contextlib.closing(self._interface):
			recommendations = self._trust_based_filterer.make_recommendations()
			self._add_recommendations(recommendations)


	def _add_recommendations(self, recommendations):

		for customer_id, products, recommendation_coefficients in recommendations:
			print(products)

	def _add_recommendation(self, customer_id, product_id, recommendation_coefficient):

		self._interface.create_relationship(
			node_labels=("Product", "Customer"),
			node_ids=(product_id, customer_id),
			relationship=("RECOMMENDEDBYREF", "right"),
			properties={
				"recommendation_coefficient": recommendation_coefficient,
			},
		)
