from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl.meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl.meta.data_processors.processor_yahoofinance import (
    YahooFinanceProcessor as YahooFinance,
)
from copy import copy


class DataProcessor:
    def __init__(self, data_source, tech_indicator=None, extra_indicator=None, vix=None, **kwargs):
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                API_BASE_URL = kwargs.get("API_BASE_URL")
                self.processor = Alpaca(API_KEY, API_SECRET, API_BASE_URL)
                print("Alpaca successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "wrds":
            self.processor = Wrds()

        elif data_source == "yahoofinance":
            self.processor = YahooFinance()

        else:
            raise ValueError("Data source input is NOT supported yet.")

        # Initialize variable in case it is using cache and does not use download_data() method
        self.tech_indicator_list = tech_indicator
        self.vix = vix
        self.extra_indicator_list = extra_indicator

    def download_data(
        self, ticker_list, start_date, end_date, time_interval
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)

        return df


    def add_technical_indicator(self, df, tech_indicator_list, extra_indicator_list=None) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        self.extra_indicator_list = extra_indicator_list

        all_indicator_list = copy(tech_indicator_list)
        if extra_indicator_list is not None:
            all_indicator_list += extra_indicator_list

        print(f"Adding indicators: {all_indicator_list}")
        df = self.processor.add_technical_indicator(df, all_indicator_list)
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def add_vixor(self, df) -> pd.DataFrame:
        df = self.processor.add_vixor(df)

        return df

    def df_to_array(self, df, if_vix=None, return_timestamps=False, **kwargs) -> np.array:
        if_vix = self.vix if if_vix is None else if_vix 
        price_array, tech_array, turbulence_array, *extra_arrays = self.processor.df_to_array(
            df,
            if_vix,
            self.tech_indicator_list,
            self.extra_indicator_list if self.extra_indicator_list else [],
            **kwargs
        )

        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        if return_timestamps: 
            # timestamp_array = df['timestamp'].unique().to_numpy() # TODO: save/load in UNIX timestamps instead of pandas dates
            timestamp_array = np.sort(df["timestamp"].unique().to_numpy())

            return price_array, tech_array, turbulence_array, timestamp_array, *extra_arrays
        else:
            return price_array, tech_array, turbulence_array, *extra_arrays
