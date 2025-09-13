from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Iterable
import pathlib
import pandas as pd
import numpy as np
import catboost as cb
import requests

# Column mapping for the GicaHack / meter export format
DEFAULT_COLMAP = {
    'Meter': 'meter',
    'Clock (8:0-0:1.0.0*255:2)': 'timestamp',
    'Active Energy Import (3:1-0:1.8.0*255:2)': 'energy_import',
    'Active Energy Export (3:1-0:2.8.0*255:2)': 'energy_export',
    'TransFullCoef': 'transform_coef'
}


def _as_tz(ts, tz: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)

def get_unified_weather_data(start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             lat: float = 47.01,
                             lon: float = 28.86,
                             tz: str = "Europe/Chisinau") -> pd.DataFrame:
    """
    Return hourly weather (temperature, humidity, pressure) with a tz-aware index in `tz`.
    Combines Open-Meteo archive (past) + forecast (future) and slices to [start_date, end_date].
    """
    # --- normalize inputs to tz-aware and hour grid ---
    start = _as_tz(start_date, tz).floor("H")
    end   = _as_tz(end_date,   tz).ceil("H")
    now   = pd.Timestamp.now(tz=tz).floor("H")

    parts = []

    # --------- Part 1: archive (past) ----------
    hist_start = start
    hist_end   = min(now - pd.Timedelta(hours=1), end)  # archive reliable up to last full hour
    if hist_start <= hist_end:
        api_url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={hist_start.date().isoformat()}&end_date={hist_end.date().isoformat()}"
            "&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
            f"&timezone={tz}"
        )
        r = requests.get(api_url, timeout=30); r.raise_for_status()
        h = r.json()["hourly"]
        df_hist = pd.DataFrame(h)
        df_hist["time"] = pd.to_datetime(df_hist["time"])
        # IMPORTANT: strings are in local clock; assign tz (do not convert)
        df_hist["time"] = df_hist["time"].dt.tz_localize(tz)
        parts.append(df_hist)

    # --------- Part 2: forecast (future) ----------
    fc_start = max(start, now)
    if fc_start <= end:
        forecast_hours = int((end - now).total_seconds() // 3600) + 1
        api_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
            f"&forecast_hours={forecast_hours}&timezone={tz}"
        )
        r = requests.get(api_url, timeout=30); r.raise_for_status()
        f = r.json()["hourly"]
        df_fc = pd.DataFrame(f)
        df_fc["time"] = pd.to_datetime(df_fc["time"])
        # IMPORTANT: assign tz (do not convert)
        df_fc["time"] = df_fc["time"].dt.tz_localize(tz)
        parts.append(df_fc)

    if not parts:
        return pd.DataFrame(columns=["temperature","humidity","pressure"]).set_index(
            pd.DatetimeIndex([], tz=tz)
        )

    # --------- Combine, clean, slice ----------
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset=["time"])
            .set_index("time")
            .sort_index()
            .rename(columns={
                "temperature_2m": "temperature",
                "relative_humidity_2m": "humidity",
                "surface_pressure": "pressure",
            }))

    # ensure hourly grid (nearest within 1h) and slice exact window
    df = df.resample("H").nearest(limit=1)
    df = df.loc[start:end]

    # fill any tiny gaps defensively
    if df.isna().any().any():
        df = df.fillna(method="ffill").fillna(method="bfill")

    return df


@dataclass
class GicaHackDataLoader:
    """Loader/processor for GicaHack style meter CSV exports.

    Responsibilities:
    - Discover CSV files in a directory
    - Load & concatenate (semicolon separated) handling empty files
    - Rename columns to canonical names (configurable)
    - Parse timestamps (day-first, format='%d.%m.%Y %H:%M:%S')
    - Sort and compute per-interval diffs for cumulative import/export
    - Clean negative/zero diffs (counter resets or duplicates)
    - Provide per-meter Series/DataFrames ready for forecasting

    Parameters
    ----------
    data_dir : path-like
        Directory containing CSV export files.
    colmap : dict, optional
        Mapping from raw column names to canonical names.
    expected_freq : str, optional
        Target frequency (e.g., '15T'). Used in reindexing helpers.
    drop_empty : bool
        Skip zero-byte files when reading.
    enforce_positive : bool
        If True, non-positive diffs will be set to NaN.
    diff_columns : tuple
        Columns considered cumulative and to be differenced.
    timestamp_format : str
        Explicit format for parsing timestamps; set None to allow auto.
    verbose : bool
        If True, prints progress info.
    """
    data_dir: Union[str, os.PathLike]
    colmap: Dict[str, str] = field(default_factory=lambda: DEFAULT_COLMAP.copy())
    expected_freq: str = '15T'
    drop_empty: bool = True
    enforce_positive: bool = False
    diff_columns: Iterable[str] = ('energy_import', 'energy_export')
    timestamp_format: Optional[str] = '%d.%m.%Y %H:%M:%S'
    verbose: bool = False

    _df: Optional[pd.DataFrame] = field(init=False, default=None, repr=False)

    # ------------------------ Public API ------------------------
    def load(self) -> 'GicaHackDataLoader':
        path = pathlib.Path(self.data_dir)
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")
        files = sorted(path.glob('*.csv'))
        if not files:
            raise FileNotFoundError(f"No CSV files in {path}")
        frames = []
        for fp in files:
            if self.drop_empty and fp.stat().st_size == 0:
                if self.verbose:
                    print(f"Skipping empty file: {fp.name}")
                continue
            try:
                df_part = pd.read_csv(fp, sep=';', engine='python')
            except Exception as e:
                raise RuntimeError(f"Failed reading {fp}: {e}")
            frames.append(df_part)
        if not frames:
            raise ValueError("No non-empty CSV content loaded.")
        raw = pd.concat(frames, ignore_index=True)
        raw = raw.drop(raw.columns[5], axis=1) # remove empty column
        if self.verbose:
            print(f"Concatenated shape: {raw.shape}")
        # Rename
        raw = raw.rename(columns=self.colmap)
        # Parse timestamp
        if 'timestamp' not in raw.columns:
            raise KeyError("Column 'timestamp' missing after renaming")
        raw['timestamp'] = pd.to_datetime(
            raw['timestamp'],
            format=self.timestamp_format,
            errors='coerce',
            dayfirst=True if self.timestamp_format else False
        )
        raw = raw.dropna(subset=['timestamp'])
        # Sort
        if 'meter' not in raw.columns:
            raise KeyError("Column 'meter' missing after renaming")
        raw = raw.sort_values(['meter', 'timestamp']).reset_index(drop=True)
        # Compute diffs
        for c in self.diff_columns:
            if c in raw.columns:
                diff_name = c.replace('energy_', '') + '_diff'
                raw[diff_name] = raw.groupby('meter')[c].diff().fillna(0)
                if self.enforce_positive:
                    raw.loc[raw[diff_name] <= 0, diff_name] = np.nan
        # Compute periods in minutes
        raw['period_min'] = raw.groupby('meter')['timestamp'].diff().dt.total_seconds() / 60
        raw['period_min'] = raw['period_min'].fillna(0)
        self._df = raw
        return self

    def get_raw(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._df.copy()

    def list_meters(self) -> List[str]:
        self._ensure_loaded()
        return sorted(self._df['meter'].dropna().astype(str).unique().tolist())

    def meter_series(self, meter: Union[str, int], column: str = 'import_diff', aggregate_duplicates: str = 'mean', reindex: bool = True) -> pd.Series:
        """Return a single-meter time series for the specified diff column.

        Parameters
        ----------
        meter : str|int
            Meter identifier.
        column : str
            Which computed diff column to extract (e.g., 'import_diff').
        aggregate_duplicates : {'mean','sum','first','last',None}
            Strategy to aggregate duplicate timestamps before reindex.
        reindex : bool
            If True, reindex to full expected_freq grid.
        """
        self._ensure_loaded()
        dfm = self._df[self._df['meter'].astype(str) == str(meter)][['timestamp', column]].dropna(subset=[column])
        if dfm.empty:
            raise ValueError(f"No data for meter {meter} and column {column}")
        dfm = dfm.set_index('timestamp').sort_index()
        if not dfm.index.is_unique and aggregate_duplicates:
            if aggregate_duplicates not in {'mean','sum','first','last'}:
                raise ValueError("aggregate_duplicates must be one of mean,sum,first,last or None")
            if aggregate_duplicates == 'mean':
                dfm = dfm.groupby(level=0).mean()
            elif aggregate_duplicates == 'sum':
                dfm = dfm.groupby(level=0).sum()
            elif aggregate_duplicates == 'first':
                dfm = dfm.groupby(level=0).first()
            elif aggregate_duplicates == 'last':
                dfm = dfm.groupby(level=0).last()
        if reindex:
            full_index = pd.date_range(dfm.index.min(), dfm.index.max(), freq=self.expected_freq)
            dfm = dfm.reindex(full_index)
            dfm.index.name = 'timestamp'
        return dfm[column]

    def gap_report(self, meter: Union[str, int], column: str = 'import_diff') -> pd.DataFrame:
        """Return DataFrame describing gap lengths in intervals for a meter.
        """
        s = self.meter_series(meter, column=column, reindex=True)
        na_mask = s.isna().astype(int)
        if na_mask.sum() == 0:
            return pd.DataFrame(columns=['start','end','length'])
        seg_id = (na_mask.diff().fillna(0) != 0).cumsum()
        gaps = []
        for gid, grp in s.to_frame('v').assign(_na=na_mask, _seg=seg_id).groupby('_seg'):
            if grp['_na'].iloc[0] == 1:
                gaps.append({
                    'start': grp.index.min(),
                    'end': grp.index.max(),
                    'length': len(grp)
                })
        return pd.DataFrame(gaps).sort_values('length', ascending=False).reset_index(drop=True)

    def stats(self) -> pd.DataFrame:
        """Basic stats of diff columns aggregated across meters."""
        self._ensure_loaded()
        diff_cols = [c for c in self._df.columns if c.endswith('_diff')]
        data = []
        for m, grp in self._df.groupby('meter'):
            entry = {'meter': m, 'rows': len(grp)}
            for c in diff_cols:
                series = grp[c].dropna()
                if series.empty:
                    continue
                entry[f'{c}_mean'] = series.mean()
                entry[f'{c}_std'] = series.std()
                entry[f'{c}_count'] = series.count()
            data.append(entry)
        return pd.DataFrame(data)

    # ------------------------ Internal ------------------------
    def _ensure_loaded(self):
        if self._df is None:
            raise RuntimeError("Data not loaded. Call load() first.")


    def preprocess_and_normalize_consumption(self) -> pd.DataFrame:
        """
        Normalizes raw smart meter data by distributing accumulated consumption
        over a consistent 15-minute frequency, then aggregates to a final
        hourly median across all meters.

        Args:
            raw_df (pd.DataFrame): DataFrame with columns ['timestamp', 'meter', 'import_diff']
                                where 'timestamp' can have variable time gaps.

        Returns:
            pd.DataFrame: A DataFrame indexed by the hour, with a single column
                        containing the median hourly energy consumption.
        """
        # Ensure timestamp column is in datetime format
        self._df['timestamp'] = pd.to_datetime(self._df['timestamp'])

        def normalize_meter_data(df_meter):
            """
            Processes data for a single meter. This function is intended to be
            used with pandas' .groupby().apply().
            """
            # Set timestamp as index and sort to ensure correct time delta calculation
            df_meter = df_meter.sort_values('timestamp').set_index('timestamp')

            # FIX: Aggregate duplicate timestamps before resampling
            if df_meter.index.has_duplicates:
                df_meter = df_meter.groupby(df_meter.index).mean()

            # 1. Calculate time difference (in minutes) between consecutive readings
            time_deltas = df_meter.index.to_series().diff().dt.total_seconds() / 60.0

            # 2. Determine how many 15-minute "bins" each reading represents
            num_bins = (time_deltas / 15.0).fillna(1).replace(0, 1)

            # 3. Normalize the consumption by the number of bins
            df_meter['normalized_diff'] = df_meter['import_diff'] / num_bins

            # 4. Resample to a consistent 15-minute frequency and forward-fill
            df_resampled = df_meter[['normalized_diff']].resample('15min').ffill()
            
            return df_resampled

        print("Step 1: Normalizing data for each meter to a 15-minute frequency...")
        df_normalized = self._df.groupby('meter').apply(normalize_meter_data)

        print("Step 2: Aggregating to final hourly median...")
        df_normalized = df_normalized.reset_index()
        
        # --- MODIFIED LOGIC STARTS HERE ---
        
        df_normalized = df_normalized.set_index('timestamp')
        
        # Step A: Discretize to hourly data for EACH meter by SUMMING the 15-min values.
        # This calculates the total consumption for each hour for each meter.
        df_hourly_per_meter = df_normalized.groupby('meter').resample('H')['normalized_diff'].sum()
        
        # Reset the index so we can group by the hourly timestamp
        df_hourly_per_meter = df_hourly_per_meter.reset_index()
        
        # Step B: Finally, get the median of these hourly consumptions across all meters.
        df_final = df_hourly_per_meter.groupby('timestamp')['normalized_diff'].median()
        
        # --- MODIFIED LOGIC ENDS HERE ---
        
        # Return as a DataFrame with a clear column name
        self._clean_df = df_final.to_frame(name='import_diff')
        return df_final.to_frame(name='import_diff')
    
    def add_weather_features(self, lat=47.01, lon=28.86, tz='Europe/Chisinau'):
        import requests
        df = pd.DataFrame(self._clean_df)  # Ensure df is a DataFrame
        """Enriches a DataFrame with historical weather data from Open-Meteo."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex.")
        
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        print(f"Fetching weather data for Chișinău from {start_date} to {end_date}...")
        
        api_url = (
            "https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}"
            "&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
            f"&timezone={tz}"
        )
        response = requests.get(api_url)
        response.raise_for_status() # Raises an exception for bad status codes
        
        weather_data = response.json()
        df_weather = pd.DataFrame(data=weather_data['hourly'])
        df_weather['time'] = pd.to_datetime(df_weather['time'])
        df_weather = df_weather.set_index('time').rename(columns={
            'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 'surface_pressure': 'pressure'
        })
        
        df_merged = df.merge(df_weather, left_index=True, right_index=True, how='left')
        self._weather_df = df_merged
        return df_merged

    # @staticmethod
    # def create_advanced_features(_df):
    #     """
    #     Создает расширенные признаки из временного ряда с исправленной логикой
    #     для категориальных переменных.
    #     """
    #     df = _df.copy()
        
    #     # --- ИСПРАВЛЕННАЯ ЛОГИКА ДЛЯ ВРЕМЕННЫХ ПРИЗНАКОВ ---
        
    #     # 1. Сначала извлекаем временные компоненты как ЧИСЛА
    #     hour = df.index.hour
    #     day_of_week = df.index.dayofweek  # 0=понедельник, 6=воскресенье
    #     month = df.index.month
        
    #     # 2. Выполняем числовое сравнение для создания 'is_weekend'
    #     is_weekend = (day_of_week >= 5).astype(int) # 1 для выходного, 0 для буднего
        
    #     # 3. Создаем остальные временные признаки
    #     df['hour'] = hour
    #     df['day_of_week'] = day_of_week
    #     df['is_weekend'] = is_weekend
    #     df['month'] = month
    #     df['season'] = (month % 12 + 3) // 3  # 1=весна, 2=лето, 3=осень, 4=зима
        
    #     # 4. Теперь, когда все вычисления завершены, преобразуем в категориальный тип
    #     # Это хорошая практика для CatBoost.
    #     for col in ['hour', 'day_of_week', 'is_weekend', 'month', 'season']:
    #         df[col] = df[col].astype('category')

    #     # Лаговые признаки (значения за предыдущие периоды)
    #     for lag in [1, 2, 3, 24, 48, 168]:  # 1-3 часа, сутки, двое суток, неделя
    #         df[f'import_lag_{lag}'] = df['import_diff'].shift(lag)
        
    #     # Скользящие статистики
    #     windows = [4, 8, 24, 72]  # 4 часа, 8 часов, сутки, трое суток
    #     for window in windows:
    #         df[f'rolling_mean_{window}'] = df['import_diff'].shift(1).rolling(window=window, min_periods=1).mean()
    #         df[f'rolling_std_{window}'] = df['import_diff'].shift(1).rolling(window=window, min_periods=1).std()
    #         df[f'rolling_max_{window}'] = df['import_diff'].shift(1).rolling(window=window, min_periods=1).max()
        
    #     # Признаки на основе погоды
    #     df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    #     df['comfort_index'] = df['temperature'] - 0.55 * (1 - df['humidity']/100) * (df['temperature'] - 14.5)
    #     # weather rolling mean, std
    #     for window in [3, 6, 12]:  # 3 часа, 6 часов, 12 часов
    #         for feature in ['temperature', 'humidity', 'pressure']:
    #             df[f'{feature}_rolling_mean_{window}'] = df[feature].shift(1).rolling(window=window, min_periods=1).mean()
    #             df[f'{feature}_rolling_std_{window}'] = df[feature].shift(1).rolling(window=window, min_periods=1).std()

    #     # Для периодических признаков используем синусоиды
    #     MAX_HIST_TEMP = 38
    #     MAX_HIST_PRESSURE = 1083
    #     df['hour_sin'] = np.sin(2 * np.pi * df['hour'].astype(int) / 24)
    #     df['hour_cos'] = np.cos(2 * np.pi * df['hour'].astype(int) / 24)
    #     df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'].astype(int) / 7)
    #     df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'].astype(int) / 7)
    #     df['month_sin'] = np.sin(2 * np.pi * df['month'].astype(int) / 12)
    #     df['month_cos'] = np.cos(2 * np.pi * df['month'].astype(int) / 12)
    #     df['temperature_sin'] = np.sin(2 * np.pi * df['temperature'] / MAX_HIST_TEMP)
    #     df['temperature_cos'] = np.cos(2 * np.pi * df['temperature'] / MAX_HIST_TEMP)
    #     df['humidity_sin'] = np.sin(2 * np.pi * df['humidity'] / 100)
    #     df['humidity_cos'] = np.cos(2 * np.pi * df['humidity'] / 100)
    #     df['pressure_sin'] = np.sin(2 * np.pi * df['pressure'] / MAX_HIST_PRESSURE)
    #     df['pressure_cos'] = np.cos(2 * np.pi * df['pressure'] / MAX_HIST_PRESSURE)
        
    #     # Производные от давления
    #     df['pressure_change'] = df['pressure'].diff().shift(1)
    #     df['pressure_trend'] = df['pressure'].shift(1).rolling(window=6, min_periods=1).mean()
        
    #     # Удаляем строки с NaN (появились из-за лагов)
    #     df = df.dropna()
        
    #     return df
    @staticmethod
    def create_advanced_features(_df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
        """
        Advanced time-series features for CatBoost.
        - dropna=True: training/validation (drop rows with NaNs in FEATURES)
        - dropna=False: forecasting (keep the last row even if target is NaN)
        """
        df = _df.copy()

        # --- time components (numeric first) ---
        hour_num       = df.index.hour
        dow_num        = df.index.dayofweek
        month_num      = df.index.month
        is_weekend_num = (dow_num >= 5).astype(int)
        season_num     = (month_num % 12 + 3) // 3  # 1..4

        # attach
        df["hour"]        = hour_num
        df["day_of_week"] = dow_num
        df["is_weekend"]  = is_weekend_num
        df["month"]       = month_num
        df["season"]      = season_num

        # cast to categorical for CatBoost AFTER saving numeric copies
        for col in ["hour", "day_of_week", "is_weekend", "month", "season"]:
            df[col] = df[col].astype("category")

        # --- lags ---
        for lag in [1, 2, 3, 24, 48, 168]:
            df[f"import_lag_{lag}"] = df["import_diff"].shift(lag)

        # --- rolling stats (use shift(1) to avoid leakage) ---
        for window in [4, 8, 24, 72]:
            s = df["import_diff"].shift(1)
            df[f"rolling_mean_{window}"] = s.rolling(window, min_periods=1).mean()
            df[f"rolling_std_{window}"]  = s.rolling(window, min_periods=1).std()
            df[f"rolling_max_{window}"]  = s.rolling(window, min_periods=1).max()

        # --- weather interactions / dynamics ---
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
        df["comfort_index"] = df["temperature"] - 0.55 * (1 - df["humidity"]/100.0) * (df["temperature"] - 14.5)

        for window in [3, 6, 12]:
            for feature in ["temperature", "humidity", "pressure"]:
                s = df[feature].shift(1)
                df[f"{feature}_rolling_mean_{window}"] = s.rolling(window, min_periods=1).mean()
                df[f"{feature}_rolling_std_{window}"]  = s.rolling(window, min_periods=1).std()

        # --- fourier for daily/weekly using NUMERIC copies (not categorical codes) ---
        df["hour_sin"]        = np.sin(2 * np.pi * hour_num / 24)
        df["hour_cos"]        = np.cos(2 * np.pi * hour_num / 24)
        df["day_of_week_sin"] = np.sin(2 * np.pi * dow_num / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * dow_num / 7)
        df["month_sin"]       = np.sin(2 * np.pi * month_num / 12)
        df["month_cos"]       = np.cos(2 * np.pi * month_num / 12)

        # --- bounded fourier for weather ---
        MAX_HIST_TEMP     = 38.0
        MAX_HIST_PRESSURE = 1083.0
        df["temperature_sin"] = np.sin(2 * np.pi * df["temperature"] / MAX_HIST_TEMP)
        df["temperature_cos"] = np.cos(2 * np.pi * df["temperature"] / MAX_HIST_TEMP)
        df["humidity_sin"]    = np.sin(2 * np.pi * df["humidity"] / 100.0)
        df["humidity_cos"]    = np.cos(2 * np.pi * df["humidity"] / 100.0)
        df["pressure_sin"]    = np.sin(2 * np.pi * df["pressure"] / MAX_HIST_PRESSURE)
        df["pressure_cos"]    = np.cos(2 * np.pi * df["pressure"] / MAX_HIST_PRESSURE)

        # --- pressure derivatives ---
        df["pressure_change"] = df["pressure"].diff().shift(1)
        df["pressure_trend"]  = df["pressure"].shift(1).rolling(window=6, min_periods=1).mean()

        if dropna:
            # drop NaNs from FEATURES ONLY; keep NaN in target for the very last row if present
            feature_cols = [c for c in df.columns if c != "import_diff"]
            df = df.dropna(subset=feature_cols)

        return df

    @staticmethod
    def predict_future(
        model: cb.CatBoostRegressor,
        raw_historical_df: pd.DataFrame,
        future_steps: int,
        feature_creation_function: callable,
        max_lag: int = 168,
        tz: str = 'Europe/Chisinau'
    ) -> pd.DataFrame:
        """
        Performs an EFFICIENT autoregressive forecast for a specified number of future steps
        using a rolling window approach to avoid re-calculating features for the entire history.

        Args:
            model: The trained CatBoost model.
            raw_historical_df: DataFrame of historical data BEFORE feature engineering.
                            Must contain the target ('import_diff') and weather columns.
            future_steps: The number of hours to forecast into the future.
            feature_creation_function: Reference to the function that creates features.
            max_lag: The largest lag used in feature engineering (e.g., 168 for a week).
                    This determines the size of the rolling window.
            tz: The timezone for all date operations.

        Returns:
            A DataFrame containing the future predictions.
        """
        print(f"Starting EFFICIENT autoregressive forecast for {future_steps} steps...")
        
        # --- FIX 1: Ensure the historical DataFrame's index is timezone-aware ---
        if raw_historical_df.index.tz is None:
            print(f"Localizing historical data to timezone: {tz}")
            raw_historical_df.index = raw_historical_df.index.tz_localize(tz)
        
        # 1. Use a "lookback window" of recent history, which is all we need.
        history_window = raw_historical_df.iloc[-max_lag:].copy()

        # 2. Get weather forecast for the entire future period
        last_timestamp = raw_historical_df.index.max()
        forecast_start_time = last_timestamp + pd.Timedelta(hours=1)
        forecast_end_time = last_timestamp + pd.Timedelta(hours=future_steps)
        
        unified_weather_df = get_unified_weather_data(
            start_date=forecast_start_time, 
            end_date=forecast_end_time,
            tz=tz
        )
        
        # --- FIX 2: Forward-fill any gaps in the weather data to prevent NaNs ---
        if unified_weather_df.isnull().values.any():
            print("Warning: Missing values found in weather data. Forward-filling.")
            unified_weather_df = unified_weather_df.ffill().bfill() # bfill for any leading NaNs


        predictions = []
        
        # 3. Iteratively predict one step at a time
        for i in range(future_steps):
            current_timestamp = last_timestamp + pd.Timedelta(hours=i + 1)
            
            # Create a shell for the next step to be predicted
            next_step_shell = pd.DataFrame(index=[current_timestamp])
            next_step_shell['import_diff'] = np.nan # Placeholder for the target
            
            # Add future weather data to the shell
            try:
                weather_now = unified_weather_df.loc[current_timestamp]
                for col in weather_now.index:
                    next_step_shell[col] = weather_now[col]
            except KeyError:
                print(f"Warning: Weather data not available for {current_timestamp}. Using last known values.")
                last_weather = history_window[['temperature', 'humidity', 'pressure']].iloc[-1]
                for col in last_weather.index:
                    next_step_shell[col] = last_weather[col]

            # Combine our rolling history window with the new shell
            combined_window = pd.concat([history_window, next_step_shell])
            
            # Generate features ONLY for this small, combined window.
            df_with_features = feature_creation_function(combined_window)
            
            # The features for our prediction are in the very last row
            features_for_pred = df_with_features.drop('import_diff', axis=1).iloc[-1:]
            
            # --- FIX 3: Add a safeguard check before prediction ---
            if features_for_pred.empty:
                raise RuntimeError(
                    f"Feature engineering resulted in an empty DataFrame at timestamp {current_timestamp}. "
                    "This is likely due to NaNs being created and then dropped. "
                    "Check for missing weather data or issues in the feature creation logic."
                )
            
            # Make the prediction
            prediction = model.predict(features_for_pred)[0]
            predictions.append(prediction)
            
            # This is the crucial step: update the history window for the next iteration.
            # First, complete the shell with the prediction we just made.
            next_step_shell['import_diff'] = prediction
            
            # Append this completed row to our history window
            history_window = pd.concat([history_window, next_step_shell])
            
            # And drop the oldest row to keep the window size constant, "rolling" it forward.
            history_window = history_window.iloc[1:]

        # 4. Create the final results DataFrame
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1), 
            periods=future_steps, 
            freq='h',
            tz=tz # Ensure the final index is also timezone-aware
        )
        df_forecast = pd.DataFrame(data={'prediction': predictions}, index=future_index)
        
        print("Forecast complete.")
        return df_forecast


