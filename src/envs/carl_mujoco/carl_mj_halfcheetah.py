from __future__ import annotations

import numpy as np

from carl.context.context_space import (
    CategoricalContextFeature,
    ContextFeature,
    UniformFloatContextFeature,
)

from envs.carl_mujoco.carl_mujoco_env import CARLMujocoEnv

class CARLMujocoHalfcheetah(CARLMujocoEnv):
    env_name: str = "HalfCheetah-v4"
    asset_path: str = "envs/mujoco/assets/half_cheetah.xml"
    metadata = {"render_modes": []}

    @staticmethod
    def get_context_features() -> dict[str, ContextFeature]:
        return {
            "gravity": UniformFloatContextFeature(
                "gravity", lower=-1000, upper=-1e-6, default_value=-9.8
            ),
            # "friction": UniformFloatContextFeature(
            #     "friction", lower=0, upper=100, default_value=1
            # ),
            # "elasticity": UniformFloatContextFeature(
            #     "elasticity", lower=0, upper=100, default_value=0
            # ),
            # "ang_damping": UniformFloatContextFeature(
            #     "ang_damping", lower=-np.inf, upper=np.inf, default_value=-0.05
            # ),
            # "viscosity": UniformFloatContextFeature(
            #     "viscosity", lower=0, upper=np.inf, default_value=0
            # ),
            "mass_torso": UniformFloatContextFeature(
                "mass_torso", lower=1e-6, upper=np.inf, default_value=6.25 #6.25020921
            ),
            "mass_bthigh": UniformFloatContextFeature(
                "mass_bthigh", lower=1e-6, upper=np.inf, default_value=1.54 #1.54351464
            ),
            "mass_bshin": UniformFloatContextFeature(
                "mass_bshin", lower=1e-6, upper=np.inf, default_value=1.58 #1.5874477
            ),
            "mass_bfoot": UniformFloatContextFeature(
                "mass_bfoot", lower=1e-6, upper=np.inf, default_value=1.09 #1.0953975
            ),
            "mass_fthigh": UniformFloatContextFeature(
                "mass_fthigh", lower=1e-6, upper=np.inf, default_value=1.43 #1.43807531
            ),
            "mass_fshin": UniformFloatContextFeature(
                "mass_fshin", lower=1e-6, upper=np.inf, default_value=1.20 #1.20083682
            ),
            "mass_ffoot": UniformFloatContextFeature(
                "mass_ffoot", lower=1e-6, upper=np.inf, default_value=0.88 #0.88451883
            ),
            # "target_distance": UniformFloatContextFeature(
            #     "target_distance", lower=0, upper=np.inf, default_value=100
            # ),
            # "target_direction": CategoricalContextFeature(
            #     "target_direction", choices=directions, default_value=1
            # ),
            # "target_radius": UniformFloatContextFeature(
            #     "target_radius", lower=0.1, upper=np.inf, default_value=5
            # ),
        }