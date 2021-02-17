from gym.envs.registration import register
from epidemioptim.environments.gym_envs.get_env import get_env

register(id='EpidemicDiscrete-v0',
         entry_point='epidemioptim.environments.gym_envs.epidemic_discrete:EpidemicDiscrete')

register(id='KoreaEpidemicDiscrete-v0',
         entry_point='epidemioptim.environments.gym_envs.korea_epidemic_discrete:KoreaEpidemicDiscrete')

