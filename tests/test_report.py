"""Report seam: a file in, a standard report directory out; README matches files."""
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from analysis import equilibrium as eq
from analysis.report import build_report
from analysis.yields import load_yield_grid
from simulation.src.run import run_simulation
from tests.test_bond_graph_analysis import monomers, ring, stack, write_traj
from tests.test_gcmc_trajectory import GC, NVT

pytestmark = pytest.mark.fast
README = Path(__file__).resolve().parents[1] / 'README.md'


def synthetic(path, seed=0, n=12):
    frames = [stack(ring(4, (6, 6)), ring(4, (20, 20)), monomers([(5, 30), (30, 5)]))] * n
    return write_traj(path, frames, np.pi / 2, seed=seed)


def test_one_command_writes_yields_checks_and_plots(tmp_path):
    a, b = synthetic(tmp_path / 'a.h5'), synthetic(tmp_path / 'b.h5', seed=1)
    out = tmp_path / 'rep'
    r = subprocess.run([sys.executable, '-m', 'analysis.report', str(a), '--outdir', str(out),
                        '--seeds', str(b), '--preassembled', str(b)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    names = {p.name for p in out.iterdir()}
    assert {'job_a_yields.npz', 'yields.json', 'equilibrium.json', 'a_snapshot_production_f00011.png',
            'a_yield_particle.png', 'a_yield_cluster.png', 'a_timeseries.png',
            'comparison.png', 'a.gif'} <= names
    assert all((out / n).stat().st_size > 0 for n in names)
    rep = eq.EquilibriumReport.load(out / 'equilibrium.json')
    assert {'seed_agreement', 'forward_backward'} <= set(rep.checks)
    d = np.load(out / 'job_a_yields.npz')
    assert d['avg_Y_square'] == pytest.approx(8 / 10)       # 8 of 10 particles in squares
    assert d['n_seeds'] == 2


def test_phase_diagram_aggregation_loads_report_outputs(tmp_path):
    p = synthetic(tmp_path / 'a.h5')
    with h5py.File(p, 'a') as f:
        f.attrs['bond_specificity'] = 2.5
    build_report(p, tmp_path / 'rep', gif=False)
    alphas, zetas, grids = load_yield_grid(tmp_path / 'rep', alphas_deg=[60, 90, 108],
                                           zetas=[1.0, 2.5])
    assert grids['square'][1, 1] == pytest.approx(0.8)
    assert grids['triangle'][1, 1] == 0.0
    assert np.isnan(grids['square'][0, 0])


def _documented(prefix):
    text = README.read_text()
    line = next(l for l in text.splitlines() if l.startswith(f'- {prefix}:'))
    return set(re.findall(r'`([A-Za-z_]+)`', line))


def _attrs(path):
    with h5py.File(path) as f:
        return set(f.attrs)


def test_readme_schema_matches_written_files(tmp_path):
    nvt, gc = tmp_path / 'n.h5', tmp_path / 'g.h5'
    run_simulation('canonical', nvt, **NVT)
    run_simulation('grand_canonical', gc, **GC)
    common = _documented('Both ensembles')
    assert _attrs(nvt) == common | _documented('Canonical only')
    assert _attrs(gc) == common | _documented('Grand canonical only')
    text = README.read_text()
    section = text[text.index('### Trajectory file (HDF5)'):text.index('### Running either')]
    with h5py.File(gc) as f:
        for name in f['production']:
            assert f'`{name}`' in section, name
        assert 'restart' in f
