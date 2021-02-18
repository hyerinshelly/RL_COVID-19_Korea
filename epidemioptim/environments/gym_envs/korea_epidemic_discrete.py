import numpy as np
import gym
from epidemioptim.environments.gym_envs.base_env import BaseEnv


class KoreaEpidemicDiscrete(BaseEnv):
    def __init__(self,
                 cost_function,
                 model,
                 simulation_horizon,
                 ratio_death_to_R=0.02,  # death ratio among people who were infected
                 time_resolution=7,
                 seed=np.random.randint(1e6)
                 ):
        """
        EpidemicDiscrete environment is based on the Epidemiological SEIRAH model from Prague et al., 2020 and on a bi-objective
        cost function (death toll and gdp recess).

        Parameters
        ----------
        cost_function: BaseCostFunction
            A cost function.
        model: BaseModel
            An epidemiological model.
        simulation_horizon: int
            Simulation horizon in days.
        ratio_death_to_R: float
            Ratio of deaths among recovered individuals.
        time_resolution: int
            In days.
        """

        # Initialize model
        self.model = model
        self.stochastic = self.model.stochastic
        self.simulation_horizon = simulation_horizon
        self.reset_same = False  # whether the next reset resets the same epidemiological model

        # Initialize cost function
        self.cost_function = cost_function
        self.nb_costs = cost_function.nb_costs
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # Initialize states
        self.state_labels = self.model.internal_states_labels + ['previous_distancing_level', 'current_distancing_level'] + \
            ['cumulative_cost_{}'.format(id_cost) for id_cost in range(self.cost_function.nb_costs)] #+ ['level_c']  # not sure
        self.label_to_id = dict(zip(self.state_labels, np.arange(len(self.state_labels))))
        self.normalization_factors = [self.model.current_internal_params['N']] * len(self.model.internal_states_labels) + \
                                     [1, 1, self.model.current_internal_params['N'], self.model.current_internal_params['N']]  # not sure also. 150?

        super().__init__(cost_function=cost_function,
                         model=model,
                         simulation_horizon=simulation_horizon,
                         dim_action=6,
                         discrete=True,
                         seed=seed)

        self.ratio_death_to_R = ratio_death_to_R
        self.time_resolution = time_resolution
        self._max_episode_steps = simulation_horizon // time_resolution
        self.history = None

        # Action modalities
        # self.level_c_splits = (7, 14, 21)  # switches between transmission rates, in days (4 stages)
        # self.level_c = 0  # index of the stage
        self.c0 = self.model.current_internal_params['c0']  # initial contact per day
        self.c1 = self.model.current_internal_params['c1']
        self.c15 = self.model.current_internal_params['c15']
        self.c2 = self.model.current_internal_params['c2']
        self.c25 = self.model.current_internal_params['c25']
        self.c3 = self.model.current_internal_params['c3']
        self.cValues = [self.c0, self.c1, self.c15, self.c2, self.c25, self.c3]  # factors of reduction for each stage
        self.cs = None

    def _compute_c(self, times_since_start):
        """
        Computes the number of contacts per day depending on the number of days since the last lock-down or since beginning of the current lock-down.

        Parameters
        ----------
        times_since_start: nd.array of ints
            Time since the start of the current distancing level, for each day.

        Returns
        -------
        list
            The values of transmission rates for each day.
        """
        cs = []
        for _ in times_since_start:
            cs.append(self.cValues[self.distancing_level])

        return cs

    def _update_previous_env_state(self):
        """
        Save previous env state.

        """
        if self.env_state is not None:
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()

    def _update_env_state(self):
        """
        Update the environment state.

        """

        # Update env state
        self.env_state_labelled = dict(zip(self.model.internal_states_labels, self.model_state))
        self.env_state_labelled.update(previous_distancing_level=self.previous_distancing_level,
                                       current_distancing_level=self.distancing_level)
                                       # level_c=self.level_c)
        # track cumulative costs in the state.
        for id_cost in range(self.nb_costs):
            self.env_state_labelled['cumulative_cost_{}'.format(id_cost)] = self.cumulative_costs[id_cost]
        assert sorted(list(self.env_state_labelled.keys())) == sorted(self.state_labels), "labels do not match"
        self.env_state = np.array([self.env_state_labelled[k] for k in self.state_labels])

        # Set previous env state to env state if first step
        if self.previous_env_state is None:
            # happens at first step
            self.previous_env_state = self.env_state.copy()
            self.previous_env_state_labelled = self.env_state_labelled.copy()

    def reset_same_model(self):
        """
        To call if you want to reset to the same model the next time you call reset.
        Will be cancelled after the first reset, it needs to be called again each time.


        """
        self.reset_same = True

    def reset(self):
        """
        Reset the environment and the tracking of data.

        Returns
        -------
        nd.array
            The initial environment state.

        """
        # initialize history of states, internal model states, actions, cost_functions, deaths
        self.history = dict(env_states=[],
                            model_states=[],
                            env_timesteps=[],
                            actions=[],
                            aggregated_costs=[],
                            costs=[],
                            distancing=[],
                            deaths=[],
                            c=[])
        # initialize time and lockdown days counter
        self.t = 0
        # self.count_lockdown = 0  # not sure
        self.count_deaths = 0
        # self.count_since_start_lockdown = 0
        # self.count_since_last_lockdown = 0
        self.count_since_start_current_distancing = 0
        # self.level_c = 0
        self.c = self.model.current_internal_params['c_fit']

        self.distancing_level = 0  # 0 no distancing, 1 level1, 15 level1.5, 2 level2, 25 level2.5, 3 level3 distancing
        self.previous_distancing_level = self.distancing_level
        self.cumulative_costs = [0 for _ in range(self.nb_costs)]

        # initialize model internal state and params
        if self.reset_same:
            self.model.reset_same_model()
            self.reset_same = False
        else:
            self.model.reset()
        self.model_state = self.model._get_current_state()

        self._update_previous_env_state()
        self._update_env_state()

        self.history['env_states'].append(self.env_state.copy())
        self.history['model_states'].append(self.model_state.copy().tolist())
        self.history['env_timesteps'].append(self.t)

        return self._normalize_env_state(self.env_state)

    def update_with_action(self, action):
        """
        Implement effect of action on transmission rate.

        Parameters
        ----------
        action: int
            Action is 0 (no lock-down) or 1 (lock-down).

        """

        # Translate actions
        self.previous_distancing_level = self.distancing_level
        previous_count_start = self.count_since_start_current_distancing

        self.jump_of = min(self.time_resolution, self.simulation_horizon - self.t)
        self.distancing_level = action

        if self.previous_distancing_level == self.distancing_level:
            self.count_since_start_current_distancing += self.jump_of
        else:
            self.count_since_start_current_distancing = self.jump_of

        # Modify model parameters based on distancing level
        since_start = np.arange(previous_count_start, self.count_since_start_current_distancing)
        self.cs = self._compute_c(times_since_start=since_start)
        self.model.current_internal_params['c_fit'] = self.c

    def step(self, action):
        """
        Traditional step function from OpenAI Gym envs. Uses the action to update the environment.

        Parameters
        ----------
        action: int
            Action is 0 (no lock-down) or 1 (lock-down).


        Returns
        -------
        state: nd.array
            New environment state.
        cost_aggregated: float
            Aggregated measure of the cost.
        done: bool
            Whether the episode is terminated.
        info: dict
            Further infos. In our case, the costs, icu capacity of the region and whether constraints are violated.

        """
        action = int(action)
        assert 0 <= action < self.dim_action

        self.update_with_action(action)
        # not sure
        # if self.lockdown_state == 1:
        #     self.count_lockdown += self.jump_of

        # Run model for jump_of steps
        model_state = [self.model_state]
        model_states = []
        for c in self.cs:
            self.model.current_internal_params['c_fit'] = c
            model_state = self.model.run_n_steps(model_state[-1], 1)
            model_states += model_state.tolist()
        self.model_state = model_state[-1]  # last internal state is the new current one
        self.t += self.jump_of

        # Update state
        self._update_previous_env_state()
        self._update_env_state()

        # Store history
        costs = [c.compute_cost(previous_state=np.atleast_2d(self.previous_env_state),
                                state=np.atleast_2d(self.env_state),
                                label_to_id=self.label_to_id,
                                action=action,
                                others=dict(jump_of=self.time_resolution))[0] for c in self.cost_function.costs]
        for i in range(len(costs)):
            self.cumulative_costs[i] += costs[i]
        n_deaths = self.cost_function.compute_deaths(previous_state=np.atleast_2d(self.previous_env_state),
                                                     state=np.atleast_2d(self.env_state),
                                                     label_to_id=self.label_to_id,
                                                     action=action)[0]

        self._update_env_state()

        self.history['actions'] += [action] * self.jump_of
        self.history['env_states'] += [self.env_state.copy()] * self.jump_of
        self.history['env_timesteps'] += list(range(self.t - self.jump_of, self.t))
        self.history['model_states'] += model_states
        self.history['distancing'] += [self.distancing_level] * self.jump_of
        self.history['deaths'] += [n_deaths / self.jump_of] * self.jump_of
        self.history['c'] += self.cs

        # Compute cost_function
        cost_aggregated, costs, over_constraints = self.cost_function.compute_cost(previous_state=self.previous_env_state,
                                                                                   state=self.env_state,
                                                                                   label_to_id=self.label_to_id,
                                                                                   action=action,
                                                                                   others=dict(jump_of=self.jump_of))
        costs = costs.flatten()

        self.history['aggregated_costs'] += [cost_aggregated / self.jump_of] * self.jump_of
        self.history['costs'] += [costs / self.jump_of for _ in range(self.jump_of)]
        self.costs = costs.copy()

        if self.t >= self.simulation_horizon:
            done = 1
        else:
            done = 0

        return self._normalize_env_state(self.env_state), cost_aggregated, done, dict(costs=costs,
                                                                                      constraints=over_constraints.flatten())

    # Utils
    def _normalize_env_state(self, env_state):
        return (env_state / np.array(self.normalization_factors)).copy()

    def _set_rew_params(self, goal):
        self.cost_function.set_goal_params(goal.copy())

    def sample_cost_function_params(self):
        return self.cost_function.sample_goal_params()

    # Format data for plotting
    def get_data(self):

        data = dict(history=self.history.copy(),
                    time_jump=1,
                    model_states_labels=self.model.internal_states_labels)
        t = self.history['env_timesteps']
        cumulative_death = [np.sum(self.history['deaths'][:i]) for i in range(len(t) - 1)]
        cumulative_eco_cost = [np.array(self.history['costs'])[:i, 1].sum() for i in range(len(t) - 1)]
        betas = [0, 0.25, 0.5, 0.75, 1]
        costs = np.array(self.history['costs'])
        aggregated = [self.cost_function.compute_aggregated_cost(costs, beta) for beta in betas]
        to_plot = [np.array(self.history['deaths']),
                   np.array(cumulative_death),
                   aggregated,
                   costs[:, 1],
                   np.array(cumulative_eco_cost),
                   np.array(self.history['c'])
                   ]
        # labels = ['New Deaths', 'Total Deaths', r'Aggregated Cost', 'New GDP Loss (B)', 'Total GDP Loss (B)', 'Transmission rate']
        # legends = [None, None, [r'$\beta = $' + str(beta) for beta in betas], None, None, None]
        labels = ['New Deaths', 'Total Deaths', r'Aggregated Cost', 'New Quarantine Loss', 'Total Quarantine Loss',
                  'Average Contact Number']
        legends = [None, None, [r'$\beta = $' + str(beta) for beta in betas], None, None, None]
        stats_run = dict(to_plot=to_plot,
                         labels=labels,
                         legends=legends)
        data['stats_run'] = stats_run
        data['title'] = 'Eco cost: {}, Death Cost: {}, Aggregated Cost: {:.2f}'.format(cumulative_eco_cost[-1],
                                                                                       int(cumulative_death[-1]),
                                                                                       np.sum(self.history['aggregated_costs']))
        return data


if __name__ == '__main__':
    from epidemioptim.utils import plot_stats
    from epidemioptim.environments.cost_functions import get_cost_function
    from epidemioptim.environments.models import get_model

    simulation_horizon = 364
    stochastic = False

    model = get_model(model_id='sqeir', params=dict(region=region,
                                                      stochastic=stochastic))

    # N_region = model.pop_sizes[region]
    # N_country = np.sum(list(model.pop_sizes.values()))
    ratio_death_to_R = 0.02

    cost_func = get_cost_function(cost_function_id='korea_multi_cost_death_economy_controllable', params=dict(ratio_death_to_R=ratio_death_to_R)
                                  )

    env = gym.make('EpidemicDiscrete-v0',
                   cost_function=cost_func,
                   model=model,
                   simulation_horizon=simulation_horizon)
    env.reset()

    actions = np.random.choice([0, 1, 2, 3, 4, 5], size=53)
    actions = np.zeros([53])
    actions[3:3+8] = 1
    t = 0
    r = 0
    done = False
    while not done:
        out = env.step(actions[t])
        t += 1
        r += out[1]
        done = out[2]
    stats = env.unwrapped.get_data()

    # plot model states
    plot_stats(t=stats['history']['env_timesteps'],
               states=np.array(stats['history']['model_states']).transpose(),
               labels=stats['model_states_labels'],
               distancing=np.array(stats['history']['distancing']),
               time_jump=stats['time_jump'])
    plot_stats(t=stats['history']['env_timesteps'][1:],
               states=stats['stats_run']['to_plot'],
               labels=stats['stats_run']['labels'],
               legends=stats['stats_run']['legends'],
               title=stats['title'],
               distancing=np.array(stats['history']['distancing']),
               time_jump=stats['time_jump'],
               show=True
               )
