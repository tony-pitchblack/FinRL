from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Optional, Sequence
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from finrl.metrics import get_account_value_metrics
import numpy as np
import pandas as pd

class MetricsLoggerCallback(DefaultCallbacks):
    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        super().__init__()

        self._env_runner_indices = env_runner_indices
        self._episode_start_position = {}
        self.metric_names = set()

    def unwrap_env(self, env):
        env = env.unwrapped
        # print(type(env))
        # (SingleAgentEnvRunner pid=127688) <class 'gymnasium.vector.sync_vector_env.SyncVectorEnv'>

        if isinstance(env, AsyncVectorEnv):
            return env
        else:
            env = env.envs[0]
            # print(type(env))
            # (SingleAgentEnvRunner pid=127688) <class 'gymnasium.wrappers.common.OrderEnforcing'>

            env = env.env
            # print(type(env))
            # (SingleAgentEnvRunner pid=127688) <class 'gymnasium.wrappers.common.PassiveEnvChecker'>

            env = env.env
            # print(type(env))
            # (SingleAgentEnvRunner pid=127688) <class 'finrl.meta.env_stock_trading.env_stocktrading.StockTradingEnv'>
        return env

    def on_episode_step(
            self,
            *,
            episode,
            env_runner,
            metrics_logger,
            env,
            env_index,
            rl_module,
            **kwargs,
        ) -> None:

        env = self.unwrap_env(env)

        if isinstance(env, AsyncVectorEnv):
            asset_values = env.get_attr('asset_memory')
            asset_values = pd.concat([pd.Series(av) for av in asset_values], axis=1).mean(axis=1)
        else:
            asset_values = env.asset_memory

        metrics = get_account_value_metrics(asset_values)

        # mode = env.mode
        for metric_name, metric_value in metrics.items():
            # metric_name = f"{mode}/{metric_name}" if mode != "" else metric_name
            episode.add_temporary_timestep_data(metric_name, metric_value)
            self.metric_names.update([metric_name])

    def on_episode_end(
            self,
            *,
            episode,
            env_runner,
            metrics_logger,
            env,
            env_index,
            rl_module,
            **kwargs,
        ) -> None:
        # print(self.metric_names)

        for metric_name in self.metric_names:
            metric_values = episode.get_temporary_timestep_data(metric_name)
            metrics_logger.log_value(
                f"{metric_name}_EMA",
                np.nanmean(np.array(metric_values)),
                reduce='mean',
                ema_coeff=0.2
            )

            metrics_logger.log_value(
                f"{metric_name}_MA",
                np.nanmean(np.array(metric_values)),
                reduce='mean',
                window=20
            )