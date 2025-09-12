import tempfile
import pandas as pd
from pathlib import Path
from energy_pred import GicaHackDataLoader

def _write_csv(path: Path, rows):
    header = [
        'Meter',
        'Clock (8:0-0:1.0.0*255:2)',
        'Active Energy Import (3:1-0:1.8.0*255:2)',
        'Active Energy Export (3:1-0:2.8.0*255:2)',
        'TransFullCoef'
    ]
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False, sep=';')

def test_loader_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        rows1 = [
            [111,'08.06.2025 06:15:00',1000,10,1],
            [111,'08.06.2025 06:30:00',1005,10,1],
            [111,'08.06.2025 06:45:00',1010,10,1],
        ]
        rows2 = [
            [111,'08.06.2025 07:00:00',1016,10,1],
            [111,'08.06.2025 07:15:00',1022,10,1],
        ]
        _write_csv(p/'part1.csv', rows1)
        _write_csv(p/'part2.csv', rows2)
        loader = GicaHackDataLoader(p, verbose=False).load()
        raw = loader.get_raw()
        assert 'import_diff' in raw.columns
        meters = loader.list_meters()
        assert meters == ['111']
        s = loader.meter_series('111')
        # Expect 5-1=4 diffs positive
        assert s.dropna().shape[0] == 4
        stats = loader.stats()
        assert not stats.empty
