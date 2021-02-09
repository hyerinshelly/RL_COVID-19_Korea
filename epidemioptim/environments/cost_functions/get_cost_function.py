from epidemioptim.environments.cost_functions.multi_cost_death_gdp_controllable import MultiCostDeathGdpControllable
from epidemioptim.environments.cost_functions.korea_multi_cost_death_economy_controllable import KoreaMultiCostDeathEconomyControllable

def get_cost_function(cost_function_id, params={}):
    if cost_function_id == 'multi_cost_death_gdp_controllable':
        return MultiCostDeathGdpControllable(**params)
    elif cost_function_id == 'korea_multi_cost_death_economy_controllable':
        return KoreaMultiCostDeathEconomyControllable(**params)
    else:
        raise NotImplementedError
