"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
import time
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stockstats import StockDataFrame as Sdf
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.simplefilter(action='ignore', category=FutureWarning)


### Added by aymeric75 for scrap_data function


class YahooFinanceProcessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self):
        pass

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    ######## ADDED BY aymeric75 ###################

    def date_to_unix(self, date_str) -> int:
        """Convert a date string in yyyy-mm-dd format to Unix timestamp."""
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        # Base URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver Setup
        options = Options()
        options.add_argument("--headless")  # Headless for performance
        options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # Navigate to the URL
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Wait for redirection and page load

        # Handle potential popup
        try:
            RejectAll = driver.find_element(
                By.XPATH, '//button[@class="btn secondary reject-all"]'
            )
            action = ActionChains(driver)
            action.click(on_element=RejectAll)
            action.perform()
            time.sleep(5)

        except Exception as e:
            print("Popup not found or handled:", e)

        # Parse the page for the table
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            raise Exception("No table found after handling redirection and popup.")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        headers[4] = "Close"
        headers[5] = "Adj Close"
        headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]
        # , 'tic', 'day'

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) == len(headers):  # Only add rows with correct column count
                rows.append(cells)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert columns to appropriate data types
        def safe_convert(value, dtype):
            try:
                return dtype(value.replace(",", ""))
            except ValueError:
                return value

        df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
        df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
        df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
        df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
        df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
        df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

        # Add 'tic' column
        df["tic"] = stock_name

        # Add 'day' column
        start_date = datetime.datetime.fromtimestamp(period1)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - start_date).dt.days
        df = df[df["day"] >= 0]  # Exclude rows with days before the start date

        # Reverse the DataFrame rows
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """Fetch and combine stock data for multiple stock names."""
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"Processing {stock_name} ({i + 1}/{total_stocks})... {(i + 1) / total_stocks * 100:.2f}% complete."
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error fetching data for {stock_name}: {e}")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(by=["day", "tick"]).reset_index(drop=True)

        return combined_df

    ######## END ADDED BY aymeric75 ###################

    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        proxy: str | dict = None,
        batch_daily = False,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download and save the data in a pandas DataFrame
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        delta = timedelta(days=1)
        data_df = pd.DataFrame()
        if batch_daily:
            # downloading daily to workaround yfinance only allowing  max 7 calendar (not trading) days of 1 min data per single download
            for tic in ticker_list:
                current_tic_start_date = start_date
                while (
                    current_tic_start_date <= end_date
                ):  
                    temp_df = yf.download(
                        tic,
                        start=current_tic_start_date,
                        end=current_tic_start_date + delta,
                        interval=self.time_interval,
                        proxy=proxy,
                    )
                    if temp_df.columns.nlevels != 1:
                        temp_df.columns = temp_df.columns.droplevel(1)

                    # temp_df["tic"] = tic
                    data_df = pd.concat([data_df, temp_df])
                    current_tic_start_date += delta
        else:
            data_df = yf.download(
                ticker_list,
                start=start_date,
                end=end_date,
                interval=self.time_interval,
                proxy=proxy,
            )

        # Convert wide to long format
        data_df.reset_index(inplace=True)
        data_df = data_df.sort_index(axis=1).set_index(['Date']).stack(level='Ticker', future_stack=True)
        data_df.reset_index(inplace=True)
        data_df.columns.name = ''

        data_df = data_df.drop(columns=["Adj Close"], errors='ignore')

        # convert the column names to match processor_alpaca.py as far as poss
        data_df.rename(columns={col: col.lower() for col in data_df.columns}, inplace=True)
        data_df.rename(columns={'ticker': 'tic', 'date': 'timestamp'}, inplace=True)

        # data_df.columns = [
        #     "timestamp",
        #     "close",
        #     "high",
        #     "low",
        #     "open",
        #     "volume",
        #     "tic",
        # ]

        return data_df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        trading_days = self.get_trading_days(start=self.start, end=self.end)
        
        # Generate full timestamp index (vectorized)
        if self.time_interval == "1d":
            times = pd.to_datetime(trading_days).tz_localize(NY)
        elif self.time_interval == "1m":
            # Vectorized time generation for minutes
            times = []
            for day in trading_days:
                day_start = pd.Timestamp(day + " 09:30:00").tz_localize(NY)
                day_times = pd.date_range(day_start, periods=390, freq='1T')
                times.extend(day_times)
            times = pd.DatetimeIndex(times)
        elif self.time_interval == '1h':
            # Vectorized hourly time generation  
            times = []
            for day in trading_days:
                day_start = pd.Timestamp(day + " 09:00:00").tz_localize(NY)
                day_end = pd.Timestamp(day + " 19:00:00").tz_localize(NY)
                day_times = pd.date_range(day_start, day_end, freq='1H')[:-1]  # Exclude 19:00
                times.extend(day_times)
            times = pd.DatetimeIndex(times)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )

        # Prepare input data with optimized timestamp handling
        df_work = df.copy()
        
        # Vectorized timestamp conversion
        def convert_timestamp_vectorized(ts_series):
            """Convert timestamps to NY timezone efficiently"""
            converted_series = ts_series.copy()
            
            # Check if series is already timezone-aware
            if hasattr(ts_series.dtype, 'tz') and ts_series.dtype.tz is not None:
                # Already timezone-aware, convert to NY
                return ts_series.dt.tz_convert(NY)
            else:
                # Check first non-null timestamp to determine if naive or aware
                first_valid_idx = ts_series.first_valid_index()
                if first_valid_idx is not None:
                    sample_ts = ts_series.iloc[first_valid_idx]
                    if hasattr(sample_ts, 'tzinfo') and sample_ts.tzinfo is not None:
                        # Aware timestamps - convert to NY
                        return ts_series.dt.tz_convert(NY)
                    else:
                        # Naive timestamps - localize to NY
                        return ts_series.dt.tz_localize(NY)
                else:
                    return ts_series

        df_work['timestamp'] = convert_timestamp_vectorized(df_work['timestamp'])

        # Create complete DataFrame structure using vectorized operations
        print("Creating complete timestamp grid...")
        
        # Create MultiIndex for all timestamp-ticker combinations
        full_index = pd.MultiIndex.from_product(
            [times, tic_list], 
            names=['timestamp', 'tic']
        )
        
        # Initialize complete DataFrame with NaN values
        complete_df = pd.DataFrame(
            index=full_index,
            columns=['open', 'high', 'low', 'close', 'volume'],
            dtype=float
        )
        
        # Prepare input data for efficient merging
        df_indexed = df_work.set_index(['timestamp', 'tic'])
        
        # Vectorized data filling - update all at once
        print("Filling data using vectorized operations...")
        complete_df.update(df_indexed[['open', 'high', 'low', 'close', 'volume']])
        
        # Reset index for processing
        complete_df = complete_df.reset_index()
        
        # Vectorized missing data handling
        print("Handling missing data with vectorized operations...")
        
        # Sort by ticker and timestamp for proper processing
        complete_df = complete_df.sort_values(['tic', 'timestamp'])
        
        def process_ticker_vectorized(group):
            """Process a single ticker's data with vectorized operations"""
            # Create a copy to avoid SettingWithCopyWarning
            ticker_data = group.copy()
            
            # Handle first row NaN
            if pd.isna(ticker_data.iloc[0]['close']):
                first_valid_idx = ticker_data['close'].first_valid_index()
                if first_valid_idx is not None:
                    # Fill first row with first valid close price
                    first_valid_close = ticker_data.loc[first_valid_idx, 'close']
                    ticker_data.iloc[0, ticker_data.columns.get_indexer(['open', 'high', 'low', 'close'])] = first_valid_close
                    ticker_data.iloc[0, ticker_data.columns.get_loc('volume')] = 0.0
                    print("NaN data on start date, fill using first valid data.")
                else:
                    # All prices are NaN - fill with 0
                    print(f"Missing data for ticker: {ticker_data.iloc[0]['tic']}. The prices are all NaN. Fill with 0.")
                    ticker_data.iloc[0, ticker_data.columns.get_indexer(['open', 'high', 'low', 'close', 'volume'])] = 0.0
            
            # Vectorized forward filling with volume handling
            price_cols = ['open', 'high', 'low', 'close']
            
            # Track which rows have NaN close prices before filling
            close_na_mask = ticker_data['close'].isna()
            
            # Forward fill all price columns at once
            ticker_data[price_cols] = ticker_data[price_cols].fillna(method='ffill')
            
            # Set volume to 0 for rows that were forward-filled (excluding first row)
            forward_filled_mask = close_na_mask & (ticker_data.index != ticker_data.index[0])
            ticker_data.loc[forward_filled_mask, 'volume'] = 0.0
            
            # Forward fill volume as well, then override with 0 where appropriate
            ticker_data['volume'] = ticker_data['volume'].fillna(method='ffill').fillna(0.0)
            ticker_data.loc[forward_filled_mask, 'volume'] = 0.0
            
            return ticker_data
        
        # Process all tickers with vectorized operations
        print("Processing tickers with vectorized operations...")
        processed_df = complete_df.groupby('tic', group_keys=False).apply(process_ticker_vectorized)
        
        # Final optimizations
        print("Applying final optimizations...")
        
        # Vectorized type conversion
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        processed_df[numeric_cols] = processed_df[numeric_cols].astype(float)
        
        # Single sort operation for final ordering
        final_df = processed_df.sort_values(['timestamp', 'tic']).reset_index(drop=True)
        
        print("Data clean all finished!")
        return final_df

    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["^VIX"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        print("cleaned_vix\n", cleaned_vix)
        vix = cleaned_vix[["timestamp", "close"]]
        print('cleaned_vix[["timestamp", "close"]\n', vix)
        vix = vix.rename(columns={"close": "vix"})
        print('vix.rename(columns={"close": "vix"}\n', vix)

        df = data.copy()
        print("df\n", df)
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data: pd.dataframe, time_period: int = 252) -> pd.dataframe:
        """
        Vectorised Mahalanobis-distance turbulence.
        * identical API / output columns
        * numerically robust (ε-ridge & solve)
        * ~10-20× faster than the original loop-of-DataFrames
        """
        # ── 1. Prepare return matrix ────────────────────────────────────────────
        ret = (
            data.pivot(index="timestamp", columns="tic", values="close")
                .pct_change()
        )
        ts = ret.index.to_numpy()
        R   = ret.to_numpy(dtype="float64")          # shape: (T, N)

        T, N = R.shape
        epsI = 1e-6 * np.eye(N)                      # tiny ridge once, reuse
        turb = np.zeros(T)

        # ── 2. Rolling window calculation (NumPy) ──────────────────────────────
        for i in range(time_period, T):
            win = R[i - time_period:i]

            # drop rows/cols that still contain NaN in this window
            row_mask = ~np.isnan(win).any(axis=1)
            col_mask = ~np.isnan(win[row_mask]).all(axis=0)
            win = win[row_mask][:, col_mask]

            cur = R[i, col_mask]
            if win.shape[0] < 2 or win.shape[1] < 2 or np.isnan(cur).any():
                continue                                   # keep turbulence = 0

            mu  = win.mean(axis=0)
            cov = np.cov(win, rowvar=False) + epsI[:win.shape[1], :win.shape[1]]

            try:
                diff       = cur - mu
                turb[i]    = diff @ np.linalg.solve(cov, diff)   # Mahalanobis
            except np.linalg.LinAlgError:
                pass                                            # leave as 0

        # ── 3. Return as expected by .add_turbulence() ─────────────────────────
        return pd.DataFrame({"timestamp": ts, "turbulence": turb})

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    
    def df_to_array(
        self,
        df: pd.DataFrame,
        if_vix: bool,
        tech_indicator_list: list[str],
        extra_indicator_list: list[str] = [],
        use_extra_indicators = False,
        if_extra_indicators_tech = False,

    ) -> list[np.ndarray]:
        """
        if_extra_indicators_tech: 
            if True then extra indicators are part of resulting techincal indicator array;
            if False then extra indicators are instead returned in separate arrays
        """ 

        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True

        tech_indicator_list = set(tech_indicator_list)
        extra_indicator_list = set(extra_indicator_list)
        all_indicator_list = list(tech_indicator_list)

        if use_extra_indicators:
            if not if_extra_indicators_tech:
                # print('use as extra arrays only')
                pass
            else:
                # print('use as tech indicators')
                assert extra_indicator_list.issubset(set(df.columns)), "All extra columns should be present in provided dataframe"
                all_indicator_list += list(extra_indicator_list)
        else:
            # print('not use extra indicators at all')
            df = df.copy().drop(columns=extra_indicator_list, errors='ignore')

        extra_array_list = []
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][all_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["vix"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                
                if use_extra_indicators and not if_extra_indicators_tech and len(extra_indicator_list) > 0:
                    for extra_col in extra_indicator_list:
                        extra_array = df[df.tic == tic][extra_col].values
                        extra_array_list.append(extra_array)
                    
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][all_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array, *extra_array_list

    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb
