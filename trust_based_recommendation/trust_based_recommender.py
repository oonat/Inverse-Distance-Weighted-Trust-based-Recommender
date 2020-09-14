import contextlib   # closing()

from distance_weighted_recommender.database import neo4j_interface
from distance_weighted_recommender.trust_based_recommendation.trust_based_filterer \
	import TrustBasedFilterer


class TrustBasedRecommender(object):

	def __init__(self):

		self._interface = neo4j_interface.Interface()

		self._trust_based_filterer = TrustBasedFilterer(
			list(self._interface.get_customer_product_pairs())
			)


	def make_recommendations(self):
		with contextlib.closing(self._interface):
			recommendations = self._trust_based_filterer.make_recommendations()
			self._add_recommendations(recommendations)


	def _add_recommendations(self, recommendations):

		for customer_id, products, recommendation_coefficients in recommendations:
			for i, recommendation_coefficient in enumerate(recommendation_coefficients):
				self._add_recommendation(
					customer_id,
					products[i],
					recommendation_coefficient,
				)

	def _add_recommendation(self, customer_id, product_id, recommendation_coefficient):

		self._interface.create_recommendation_relationship(
			customer_id=customer_id, 
			product_id=product_id, 
			recommendation_coefficient=recommendation_coefficient,
			)
