from __future__ import annotations

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv

from finrl.config import TRAINED_MODEL_DIR
from finrl.config import DATA_SAVE_DIR
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import CACHE_DIR
from finrl.main import check_and_make_directories

from finrl.utils.benchmarking import stable_hash
from pathlib import Path
import os
import pandas as pd


def test(
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
    # import data processor
    from finrl.meta.data_processor import DataProcessor

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

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    print("price_array: ", len(price_array))

    info = {
        "data_shape": data.shape,
        "num_stocks": len(ticker_list)
    }

    if drl_lib == "elegantrl":
        from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl

        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
        )
        return episode_total_assets, info
    elif drl_lib == "rllib":
        from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib

        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=cwd,
        )
        return episode_total_assets, info
    elif drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        return episode_total_assets, info
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

def build_parser():
    import argparse
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
        kwargs = {}
        account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="elegantrl",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}",
            net_dimension=512,
            kwargs=kwargs,
        )
    elif args.drl_lib == "rllib":
        import ray
        ray.shutdown()
        account_value_rllib = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="rllib",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}/checkpoint_0000{args.episode_num}/checkpoint-{args.episode_num}",
            rllib_params=RLlib_PARAMS,
        )
    elif args.drl_lib == "stable_baselines3":
        account_value_sb3 = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval=args.time_interval,
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name=args.model_name,
            cwd=f"./test_{args.model_name}.zip",
        )