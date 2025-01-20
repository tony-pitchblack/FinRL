from __future__ import annotations

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SB3_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE

from finrl.config import TRAINED_MODEL_DIR
from finrl.config import DATA_SAVE_DIR
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import CACHE_DIR
from finrl.main import check_and_make_directories

from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from utils import benchmark_exec_time
from utils import stable_hash

import pandas as pd
from pathlib import Path
import os

@benchmark_exec_time
def train(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):

    data_hash = stable_hash(tuple(sorted(ticker_list) + sorted(technical_indicator_list)))
    file_path = Path(CACHE_DIR) / f"{start_date}_{end_date}_{time_interval}_{data_hash}.csv"
    dp = DataProcessor(data_source, tech_indicator=technical_indicator_list, vix=if_vix, **kwargs)
    if os.path.isfile(file_path):
        print(f"Using cached data: {file_path}")
        data = pd.read_csv(file_path, index_col=0)
    else:
        print("Creating new data.")
        data = dp.download_data(ticker_list, start_date, end_date, time_interval)
        data = dp.clean_data(data)
        data = dp.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = dp.add_vix(data)
        data.to_csv(file_path)

    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }
    env_instance = env(config=env_config)

    # read parameters
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        break_step = kwargs.get("break_step", 1e6)
        erl_params = kwargs.get("erl_params")
        agent = DRLAgent_erl(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(
            model=model, cwd=cwd, total_timesteps=break_step
        )
    elif drl_lib == "rllib":
        total_episodes = kwargs.get("total_episodes", 100)
        rllib_params = kwargs.get("rllib_params")
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        agent_rllib = DRLAgent_rllib(
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
        )
        model, model_config = agent_rllib.get_model(model_name)
        model_config["lr"] = rllib_params["lr"]
        model_config["train_batch_size"] = rllib_params["train_batch_size"]
        model_config["gamma"] = rllib_params["gamma"]
        # ray.shutdown()
        trained_model = agent_rllib.train_model(
            model=model,
            model_name=model_name,
            model_config=model_config,
            total_episodes=total_episodes,
        )
        trained_model.save(cwd)
    elif drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 1e6)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
    
    info = {
        "data_shape": data.shape,
        "num_stocks": len(ticker_list)
    }

    return (info,)

import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--drl_lib',
        choices=['elegantrl', 'rllib', 'stable_baselines3'],
        default='stable_baselines3',
        help="Choose the DRL library for training (default: stable_baselines3)."
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='ppo',
        help="The name of the model to use. (default: ppo)"
    )
    parser.add_argument(
        '--break_step',
        type=int,
        default=int(1e5),
        help="Number of steps to break at for training (default: 1e5)."
    )
    parser.add_argument(
        '--total_episodes',
        type=int,
        default=30,
        help="Total number of episodes for training (default: 30)."
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=int(1e4),
        help="Total number of timesteps for training (default: 1e4)."
    )
    parser.add_argument(
        '--time_interval',
        type=str,
        default="1D",
        help="Time interval for the data, e.g., '1D', '1H', '5T' (default: '1D')."
    )

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, CACHE_DIR]
    )

    env = StockTradingEnv

    if args.drl_lib == "elegantrl":
        # demo for elegantrl
        kwargs = (
            {}
        )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}",
            erl_params=ERL_PARAMS,
            break_step=args.break_step, # default: 1e5
            kwargs=kwargs,
        )
    
    elif args.drl_lib == "rllib":
        # demo for rllib
        import ray
        ray.shutdown()  # always shutdown previous session if any
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="rllib",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}",
            rllib_params=RLlib_PARAMS,
            total_episodes=args.total_episodes, # default: 30
        )

    elif args.drl_lib == "stable_baselines3":
        # demo for stable-baselines3
        train(
            start_date=TRAIN_START_DATE,
            end_date=TRAIN_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}",
            agent_params=SB3_PARAMS[args.model_name],
            total_timesteps=args.total_timesteps, # default: 1e4
        )