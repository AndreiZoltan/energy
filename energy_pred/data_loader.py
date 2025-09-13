from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Iterable
import pathlib
import pandas as pd
import numpy as np

# Column mapping for the GicaHack / meter export format
DEFAULT_COLMAP = {
    'Meter': 'meter',
    'Clock (8:0-0:1.0.0*255:2)': 'timestamp',
    'Active Energy Import (3:1-0:1.8.0*255:2)': 'energy_import',
    'Active Energy Export (3:1-0:2.8.0*255:2)': 'energy_export',
    'TransFullCoef': 'transform_coef'
}

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


    def get_median_df(self) -> pd.DataFrame:
        df_2025 = self._df.copy()
        df_2025 = df_2025.set_index('timestamp')
        df_2025 = df_2025.groupby('meter').resample('H').median()
        df_2025 = df_2025.reset_index(level='timestamp')
        df_2025 = df_2025.set_index('timestamp')
        df_2025 = df_2025.groupby('timestamp')['import_diff'].median()
        return df_2025
