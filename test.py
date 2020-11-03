from test_module.test_algo import Inverse_distance_weighted_tbr
from surprise import Dataset
from surprise.model_selection import cross_validate


data = Dataset.load_builtin('ml-100k')

sim_options = {'name': 'cosine',
               }

algo = Inverse_distance_weighted_tbr(sim_options=sim_options)

cross_validate(algo, data, cv=5, verbose=True)