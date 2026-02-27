from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from rl_zoo3.utils import get_callback_list, get_wrapper_class, linear_schedule
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import constant_fn
import yaml

# Based on function read_hyperparameters in rl-baselines3-zoo/rl_zoo3/exp_manager.py
def read_hyperparams(
    config_path:str,
    gym_id:str,
) -> Dict[str, Any]:
    # config_path = '/home/param/crl_notebooks/hyperparams/dqn.yml'
    # algo = 'dqn'
    # gym_id = carl_env_fn.env_name
    if config_path.endswith(".yml") or config_path.endswith(".yaml"):
        with open(config_path) as f:
            hyperparams_dict = yaml.safe_load(f)
    
    if gym_id in list(hyperparams_dict.keys()):
        unprocessed_hyperparams = hyperparams_dict[gym_id]
    else:
        key_list = list(hyperparams_dict.keys())
        # lower used for case insensitive comparison
        close_ids_list = [i for i in key_list if gym_id.lower() in i.lower()]
        if len(close_ids_list)>0:
            unprocessed_hyperparams = hyperparams_dict[close_ids_list[0]]
            print(
                f'Warning: Hyperparameters not found for {gym_id} in {config_path}. '
                f'Using hyperparameters for the close_id {close_ids_list[0]}.'
            )
        else:
            # print('hyperparams_dict.keys() =', hyperparams_dict.keys())
            raise ValueError(f"Hyperparameters not found for {gym_id} in {config_path}")
    
    # print('unprocessed_hyperparams = ', unprocessed_hyperparams)
    return unprocessed_hyperparams

def preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    # Create schedules
    for key in ["learning_rate", "clip_range", "clip_range_vf", "delta_std"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams

def preprocess_hyperparams(  # noqa: C901
        input_hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback], Optional[Callable]]:
    hyperparams = input_hyperparams
    # self.n_envs = hyperparams.get("n_envs", 1)

    # if self.verbose > 0:
    #     print(f"Using {self.n_envs} environments")

    # Convert schedule strings to objects
    hyperparams = preprocess_schedules(hyperparams)

    # Pre-process train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # Should we overwrite the number of timesteps?
    # if self.n_timesteps > 0:
    #     if self.verbose:
    #         print(f"Overwriting n_timesteps with n={self.n_timesteps}")
    # else:
    #     self.n_timesteps = int(hyperparams["n_timesteps"])

    # Derive n_evaluations from number of timesteps if needed
    # if self.n_evaluations is None and self.optimize_hyperparameters:
    #     self.n_evaluations = max(1, self.n_timesteps // int(1e5))
    #     print(
    #         f"Doing {self.n_evaluations} intermediate evaluations for pruning based on the number of timesteps."
    #         " (1 evaluation every 100k timesteps)"
    #     )

    # Pre-process normalize config
    # hyperparams = preprocess_normalization(hyperparams)
    if "normalize" in hyperparams.keys():
        del hyperparams["normalize"]

    # Pre-process policy/buffer keyword arguments
    # Convert to python object if needed
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    # Preprocess monitor kwargs
    if "monitor_kwargs" in hyperparams.keys():
        # self.monitor_kwargs = hyperparams["monitor_kwargs"]
        # # Convert str to python code
        # if isinstance(self.monitor_kwargs, str):
        #     self.monitor_kwargs = eval(self.monitor_kwargs)
        del hyperparams["monitor_kwargs"]

    # Delete keys so the dict can be pass to the model constructor
    if "n_envs" in hyperparams.keys():
        del hyperparams["n_envs"]
    del hyperparams["n_timesteps"]

    if "frame_stack" in hyperparams.keys():
        # self.frame_stack = hyperparams["frame_stack"]
        del hyperparams["frame_stack"]

    # import the policy when using a custom policy
    if "policy" in hyperparams and "." in hyperparams["policy"]:
        hyperparams["policy"] = get_class_by_name(hyperparams["policy"])

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    # Same for VecEnvWrapper
    vec_env_wrapper = get_wrapper_class(hyperparams, "vec_env_wrapper")
    if "vec_env_wrapper" in hyperparams.keys():
        del hyperparams["vec_env_wrapper"]

    callbacks = get_callback_list(hyperparams)
    if "callback" in hyperparams.keys():
        # self.specified_callbacks = hyperparams["callback"]
        del hyperparams["callback"]

    return hyperparams, env_wrapper, callbacks, vec_env_wrapper