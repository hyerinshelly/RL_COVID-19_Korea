from epidemioptim.environments.cost_functions.costs.base_cost_function import BaseCostFunction

# import numpy as np


class KoreaEconomy(BaseCostFunction):
    def __init__(self,
                 id_cost,
                 ratio_death_to_R,
                 scale_factor=0.65 * 1e3, # not sure of the meaning
                 range_constraints=(1000, 62000)):
        """
        Economic cost computed as GDP recess due to diseased and dead people, as well as partial unemployment due to lock-downs.
        GDP is expressed in billions.

        Parameters
        ----------
        id_cost: int
            Identifier of the cost in the list of costs
        ratio_death_to_R: float
            Ratio of dead people computed from the number of recovered people, (in [0, 1]).
        scale_factor: float
            Scaling factor of the cost (in [0, 1])
        range_constraints: tuple
            Min and max values for the maximum constraint on the cost (size 2).

        Attributes
        ----------
        ratio_death_to_R
        id_cost
        """
        super().__init__(scale_factor=scale_factor,
                         range_constraints=range_constraints)

        self.ratio_death_to_R = ratio_death_to_R
        self.id_cost = id_cost
        # self.N_region = N_region
        # self.N_country = N_country
        # self._ratio_pop = self.N_region / self.N_country

        # Economic model parameters
        # self._N0 = self.N_region  # initial population
        # self._L0 = 25 * 1e6 * self._ratio_pop  # initial workforce # initial workforce
        # self._lambda_0 = self._L0 / self._N0  # initial activity ratio (computed)
        # self._alpha = 0.37  # elasticity wrt capital
        # self._K0 = 7481400 * 1e6 * self._ratio_pop  # initial capital (prorata region pop)
        # self._Y0 = 2317000 * 1e6 * self._ratio_pop  # initial GDP
        # self._A = self._Y0 / (self._K0 ** self._alpha * (self._lambda_0 * self._N0) ** (1 - self._alpha))  # exogeneous technical progress
        # self._A = self._A / 365  # normalize to compute GDP in a day
        # self._u = dict(zip([0, 1], [0, 0.5]))

    def compute_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Computes GDP loss since the last state.

        Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        time_jump: int
            Number of days in this setting. This scales the GDP loss per day.

        Returns
        -------
        lockdown_cost: 1D nd.array
            cost of lockdowns for each state.

        """
        quarantined = state[:, label_to_id['S_q']] + state[:, label_to_id['E_q']] + state[:, label_to_id['I_q']] + self.ratio_death_to_R * state[:, label_to_id['R']]

        return quarantined

    def compute_cumulative_cost(self, previous_state, state, label_to_id, action, others={}):
        """
        Compute cumulative costs since start of episode.

        Parameters
        ----------
               Parameters
        ----------
        previous_state: 2D nd.array
            Previous model states (either 1D or 2D with first dimension # of states).
        state: 2D nd.array
            Current model states (either 1D or 2D with first dimension # of states).
        label_to_id: dict
            Mapping between state labels and indices in the state vector.
        time_jump: int
            Number of days in this setting. This scales the GDP loss per day.

        Returns
        -------
        cumulative_cost: 1D nd.array
            Cumulative costs for each state.
        """
        cumulative_cost = state[:, label_to_id['cumulative_cost_{}'.format(self.id_cost)]]
        return cumulative_cost
