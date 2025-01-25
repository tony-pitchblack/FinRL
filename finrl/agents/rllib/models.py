# DRL models from RLlib
from __future__ import annotations

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

from finrl.agents.rllib.callbacks import MetricsLoggerCallback
from finrl.agents.rllib.utils import print_result

MODEL_CONFIGS = {
    'ppo': PPOConfig,
    'sac': SACConfig,
}

class DRLAgent:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
        price_array: numpy array
            OHLC data
        tech_array: numpy array
            techical data
        turbulence_array: numpy array
            turbulence/risk data
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, price_array, tech_array, turbulence_array, val_env_config):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.turbulence_array = turbulence_array
        self.val_env_config = val_env_config

    def get_model(
        self,
        model_name,
        model_kwargs
    ):
        if model_name not in MODEL_CONFIGS:
            raise NotImplementedError("NotImplementedError")

        model_config = (
            MODEL_CONFIGS[model_name]()
            .env_runners(
                num_env_runners=0,
            )
            .environment(
                env=self.env,
                env_config={
                    "price_array": self.price_array,
                    "tech_array": self.tech_array,
                    "turbulence_array": self.turbulence_array,
                    "if_train": True,
                },
            )
            .training(
                **model_kwargs
            )
            .evaluation(
                evaluation_interval=1,  # Specify evaluation frequency (1=after each training step)
                evaluation_config={
                    'env': self.env,
                    'env_config': self.val_env_config
                },
            )
            .callbacks(MetricsLoggerCallback)
        )

        model = model_config.build()

        return model

    def train_model(
        self, model, total_episodes=100, init_ray=True
    ):
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

        for ep_idx in range(total_episodes):
            print(f"Training episode: {ep_idx + 1}/{total_episodes}")
            result = model.train()

        print_result(result)

        ray.shutdown()

        return model

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        price_array,
        tech_array,
        turbulence_array,
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_name == "a2c":
            model_config = MODELS[model_name].A2C_DEFAULT_CONFIG.copy()
        elif model_name == "td3":
            model_config = MODELS[model_name].TD3_DEFAULT_CONFIG.copy()
        else:
            model_config = MODELS[model_name].DEFAULT_CONFIG.copy()
        model_config["env"] = env
        model_config["log_level"] = "WARN"
        model_config["env_config"] = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_config = {
            "price_array": price_array,
            "tech_array": tech_array,
            "turbulence_array": turbulence_array,
            "if_train": False,
        }
        env_instance = env(config=env_config)

        # ray.init() # Other Ray APIs will not work until `ray.init()` is called.
        if model_name == "ppo":
            trainer = MODELS[model_name].PPOTrainer(env=env, config=model_config)
        elif model_name == "a2c":
            trainer = MODELS[model_name].A2CTrainer(env=env, config=model_config)
        elif model_name == "ddpg":
            trainer = MODELS[model_name].DDPGTrainer(env=env, config=model_config)
        elif model_name == "td3":
            trainer = MODELS[model_name].TD3Trainer(env=env, config=model_config)
        elif model_name == "sac":
            trainer = MODELS[model_name].SACTrainer(env=env, config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        # test on the testing env
        state = env_instance.reset()
        episode_returns = []  # the cumulative_return / initial_account
        episode_total_assets = [env_instance.initial_total_asset]
        done = False
        while not done:
            action = trainer.compute_single_action(state)
            state, reward, done, _ = env_instance.step(action)

            total_asset = (
                env_instance.amount
                + (env_instance.price_ary[env_instance.day] * env_instance.stocks).sum()
            )
            episode_total_assets.append(total_asset)
            episode_return = total_asset / env_instance.initial_total_asset
            episode_returns.append(episode_return)
        ray.shutdown()
        print("episode return: " + str(episode_return))
        print("Test Finished!")
        return episode_total_assets
