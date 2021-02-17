# This model is an implementation of:
# Population modeling of early COVID-19 epidemic dynamics in French regions and estimation of the lockdown impact on infection rate
# Prague et al., 2020

from scipy.integrate import odeint
import pandas as pd

from epidemioptim.environments.models.base_model import BaseModel
from epidemioptim.utils import *

PATH_TO_FITTED_PARAMS = get_repo_path() + '/data/model_data/estimatedIndividualParameters.csv'
PATH_TO_FITTED_COV = get_repo_path() + '/data/model_data/data_cov.csv'

# ODE model - SEIRAH model
def sqeir_model(y: tuple,
                t: int,
                De: float,
                Di: float,
                Dt: float,
                N: int,
                beta: float,
                c: float,
                q: float):
    """ XXXXX NEED TO BE FIXED XXXXX
    SEIRAH epidemiological model from Population modeling of early COVID-19 epidemic dynamics in French
    regions and estimation of the lockdown impact on infection rate, Prague et al., 2020.
    Parameters
    ----------
    y: tuple
       Current states SEIRAH.
       y = [S, E, I, R, A, H]
       S: # susceptible individuals
       E: # individuals in latent state
       I: # symptomatic infected individuals
       R: # recovered & dead individuals (deaths represent 0.5 % of R).
       A: # asymptomatic infected individuals
       H: # hospitalized individuals
    t: int
       Timestep.
    De: float
        Latent period (days).
    Dh: float
        Average duration of hospitalizations (days).
    Di: float
        Infectious period (days).
    Dq: float
        Duration from illness onset to hospital (days).
    N: int
        Population size.
    alpha: float
        Ratio between transmission rates of reported vs not-reported, in [0, 1].
    b: float
       Transmission rate.
    r: float
       Verification rate

    Returns
    -------
    tuple
        Next states.
    """
    S, S_q, E, E_q, I, I_q, R = y
    dSdt = - beta*c*S*I/N - (1-beta)*c*q*S*I + S_q/14

    dS_qdt = - S_q/14 + (1-beta)*c*q*S*I

    dEdt = - (1-q)*beta*c*S*I/N - E/De

    dE_qdt = - q*beta*c*S*I/N - E_q/De

    dIdt = E/De -I/Dt - I/Di

    dI_qdt = E/De + I/Dt - I_q/Di

    dRdt = (I + I_q) / Di

    dydt = [dSdt, dS_qdt, dEdt, dE_qdt, dIdt, dI_qdt, dRdt]

    return dydt


class SqeirModel(BaseModel):
    def __init__(self,
                 stochastic=False,
                 noise_params=0.1,
                 range_delay=(0, 21)
                 ):
        """
        Implementation of the SEIRAH model from Prague et al., 2020:
        Population modeling of early COVID-19 epidemic dynamics in French regions and estimation of the lockdown impact on infection rate.

        Parameters
        ----------
        region: str
                Region identifier.
        stochastic: bool
                    Whether to use stochastic models or not.
        noise_params: float
                      Normally distributed parameters have an stdev of 'noise_params' x their mean.

        Attributes
        ---------
        region
        stochastic
        noise_params
        """
        #self.fitted_params = pd.read_csv(PATH_TO_FITTED_PARAMS)
        #self.fitted_cov = pd.read_csv(PATH_TO_FITTED_COV)
        #self._regions = list(self.fitted_params['id'])
        #self.pop_sizes = dict(zip(self.fitted_params['id'], (self.fitted_params['popsize'])))
        #assert region in self._regions, 'region code should be one of ' + str(self._regions)

        #self.region = region
        self.stochastic = stochastic
        self.noise = noise_params
        self._all_internal_params_distribs = dict()
        self._all_initial_state_distribs = dict()

        # Initialize distributions of parameters and initial conditions for all regions
        self.define_params_and_initial_state_distributions()

        # Sample initial conditions and initial model parameters
        internal_params_labels = list(self._all_internal_params_distribs.keys())
        internal_params_labels.remove('c0')  # remove irrelevant keys
        internal_params_labels.remove('c1')
        internal_params_labels.remove('c15')
        internal_params_labels.remove('c2')
        internal_params_labels.remove('c25')
        internal_params_labels.remove('c3')

        # Define ODE SQEIR model
        self.internal_model = sqeir_model

        super().__init__(internal_states_labels=['S', 'S_q', 'E', 'E_q', 'I', 'I_q', 'R'],
                         internal_params_labels=internal_params_labels,
                         stochastic=stochastic,
                         range_delay=range_delay)



    def define_params_and_initial_state_distributions(self):
        """
        Extract and define distributions of parameters for all French regions
        """

        #label2ind = dict(zip(list(self.fitted_cov.columns), np.arange(len(self.fitted_cov.columns))))
        #for i in self.fitted_params.index:
            #r = self.fitted_params['id'][i]  # region
        self._all_internal_params_distribs = dict(b=LogNormalDist(params=[0.027, 0.027*self.noise], stochastic=self.stochastic),
                                                  N=DiracDist(params=51844627, stochastic=self.stochastic),
                                                  q=LogNormalDist(params=[0.8, 0.8*self.noise], stochastic=self.stochastic),
                                                  De=NormalDist(params=[5.2, 5.2 * self.noise], stochastic=self.stochastic),
                                                  Dt=NormalDist(params=[2.5, 2.5 * self.noise], stochastic=self.stochastic),
                                                  Di=NormalDist(params=[20.1, 20.1 * self.noise], stochastic=self.stochastic),
                                                  c_fit=NormalDist(params=[40, 40*self.noise], stochastic=self.stochastic),
                                                  c0=NormalDist(params=[40, 40*self.noise], stochastic=self.stochastic),
                                                  c1=NormalDist(params=[25, 25*self.noise], stochastic=self.stochastic),
                                                  c15=NormalDist(params=[20, 20 * self.noise], stochastic=self.stochastic),
                                                  c2=NormalDist(params=[15, 15 * self.noise], stochastic=self.stochastic),
                                                  c25=NormalDist(params=[10, 10 * self.noise], stochastic=self.stochastic),
                                                  c3=NormalDist(params=[5, 5 * self.noise], stochastic=self.stochastic),
                                                  )
        self._all_initial_state_distribs = dict(E0=LogNormalDist(params=[69, 69 * self.noise], stochastic=self.stochastic),
                                                I0=DiracDist(params=2, stochastic=self.stochastic),
                                                R0=DiracDist(params=0, stochastic=self.stochastic),
                                                S_q0=DiracDist(params=0, stochastic=self.stochastic),
                                                E_q0 = DiracDist(params=0, stochastic=self.stochastic),
                                                I_q0 = DiracDist(params=0, stochastic=self.stochastic),
                                                )

    def _sample_initial_state(self):
        """
        Samples an initial model state from its distribution (Dirac distributions if self.stochastic is False).


        """
        self.initial_state = dict()
        for k in self._all_initial_state_distribs.keys():
            self.initial_state[k] = self._all_initial_state_distribs[k].sample()

        # A0 is computed as a function of I0 and r_fit (see Prague et al., 2020)
        #self.initial_state['A0'] = self.initial_state['I0'] * (1 - self.current_internal_params['r_fit']) / self.current_internal_params['r_fit']
        for k in self._all_initial_state_distribs.keys():
            self.initial_state[k] = int(self.initial_state[k])

        # S0 is computed from other states, as the sum of all states equals the population size N
        self.initial_state['S0'] = self.current_internal_params['N'] - np.sum([self.initial_state['{}0'.format(s)] for s in self.internal_states_labels[1:]])

    def _sample_model_params(self):
        """
        Samples parameters of the model from their distribution (Dirac distributions if self.stochastic is False).

        """
        self.initial_internal_params = dict()
        for k in self._all_internal_params_distribs.keys():
            self.initial_internal_params[k] = self._all_internal_params_distribs[k].sample()
        self._reset_model_params()

    def run_n_steps(self, current_state=None, n=1, labelled_states=False):
        """
        Runs the model for n steps

        Parameters
        ----------
        current_state: 1D nd.array
                       Current model state.
        n: int
           Number of steps the model should be run for.

        labelled_states: bool
                         Whether the result should be a dict with state labels or a nd array.

        Returns
        -------
        dict or 2D nd.array
            Returns a dict if labelled_states is True, where keys are state labels.
            Returns an array of size (n, n_states) of the last n model states.

        """
        if current_state is None:
            current_state = self._get_current_state()

        # Use the odeint library to run the ODE model
        # print(current_state.dtype)  # object - should be float64
        # print(current_state)
        current_state = pd.to_numeric(current_state)
        # print(current_state.dtype)
        z = odeint(self.internal_model, current_state, np.linspace(0, n, n + 1), args=self._get_model_params())
        self._set_current_state(current_state=z[-1].copy())  # save new current state

        # format results
        if labelled_states:
            return self._convert_to_labelled_states(np.atleast_2d(z[1:]))
        else:
            return np.atleast_2d(z[1:])


if __name__ == '__main__':
    # Get model
    model = SqeirModel(stochastic=False)

    # Run simulation
    simulation_horizon = 364
    model_states = model.run_n_steps(n=simulation_horizon)

    # Plot
    time = np.arange(simulation_horizon)
    labels = model.internal_states_labels

    plot_stats(t=time,
               states=model_states.transpose(),
               labels=labels,
               show=True)
