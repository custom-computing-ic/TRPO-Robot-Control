import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ArmDOF_0-v0',
    entry_point='armDOF_0.envs:ArmDOF_0Env',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 150},
)
