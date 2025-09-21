from gymnasium.envs.registration import register

register(
    id="gym_libero/LiberoEnv-v0",
    entry_point="gym_libero.gymnasium_wrapper:make_libero_gymnasium_env",
)