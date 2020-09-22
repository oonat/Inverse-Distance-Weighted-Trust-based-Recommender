import numpy as np
from test.trust_based_filterer import TrustBasedFilterer

from surprise import AlgoBase, PredictionImpossible, Dataset
from surprise.model_selection import cross_validate


class Distance_weighted_tbr(AlgoBase):

	def __init__(self):

		AlgoBase.__init__(self)


	def fit(self, trainset):

		AlgoBase.fit(self, trainset)

		self._filterer = TrustBasedFilterer(list(trainset.all_ratings()))


	def estimate(self, u, i):

		if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
			raise PredictionImpossible('User and/or item is unknown.')

		indexes = np.nonzero(self._filterer._bool_table[:, i])
		rate_array = self._filterer._customers_versus_products_table[indexes, i][0]

		weight_array = self._filterer._weight_matrix[u, indexes][0]
		numerator = np.sum(rate_array * weight_array)
		denominator = np.sum(weight_array)
		estimation = 0 if denominator == 0 else float(numerator)/denominator 

		return estimation


data = data = Dataset.load_builtin('ml-100k')
algo = Distance_weighted_tbr()

cross_validate(algo, data, cv=5, verbose=True)