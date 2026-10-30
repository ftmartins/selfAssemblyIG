"""
HDF5 trajectory store shared by both ensembles (spec #1, ADR-0001/0002).

Small interface
---------------
Writer  : ``TrajectoryWriter(path, attrs, resume=False, precision='float64')``
          ``append_frame``, ``write_restart``, ``set_attrs``, ``flush``, ``close``
Reader  : ``open_trajectory(path)`` -> ``TrajectoryReader`` (HDF5) or
          ``LegacyReader`` (``.npz`` / pickled ``.npy``), same interface:
          ``attrs``, ``n_frames(phase)``, ``frames(phase, start, stop, step)``,
          ``series(phase, name)``, ``read_restart()``.

File schema (version 1)
-----------------------
Root attributes : caller-supplied run metadata (ensemble, seed, box_size, kT,
    dt, thermostat, opening_angle, E_AA, E_BB, E_AB, step counts, ...) plus
    the model constants (alpha, patch_radius, morse_r_onset, morse_r_cutoff,
    center_radius), ``schema_version``, ``software_version``, ``precision``.
Groups ``/equilibration`` and ``/production``, per-frame datasets (length F):
    ``step`` int64 (MD step or sweep within the phase), ``N`` int64 (particles
    in the frame), ``offset`` int64 (first row of the frame in the flat
    arrays), ``potential_energy`` f8, ``kinetic_energy`` f8 (NaN if unknown);
    flat particle datasets (length sum(N)): ``positions`` (P, 2) and
    ``orientations`` (P,) in the ``precision`` dtype.  Frame i is
    ``[offset[i], offset[i] + N[i])``; there is no padding.  Per-frame arrays
    are chunked by ``chunk_frames`` frames and gzip-compressed.
Group ``/restart`` (absent until the first checkpoint): attributes ``phase``
    ('equilibration' | 'production' | 'done'), ``phase_step`` and any extra
    caller attributes; datasets ``leaf_0 ... leaf_k`` (flattened MD state).
    After a run completes, phase is 'done' and it holds the final state.

Resume: opening with ``resume=True`` drops every frame written after the last
``write_restart`` (those are recomputed), so a resumed run never duplicates
frames.  ``flush`` after each chunk keeps a killed run readable.
"""
import os
import subprocess
from dataclasses import dataclass

import h5py
import numpy as np

from base import config

SCHEMA_VERSION = 1
PHASES = ('equilibration', 'production')
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PER_FRAME = [('step', 'i8'), ('N', 'i8'), ('offset', 'i8'),
              ('potential_energy', 'f8'), ('kinetic_energy', 'f8')]


def software_version():
    try:
        out = subprocess.run(['git', '-C', _REPO, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def model_constants():
    return dict(alpha=config.ALPHA, patch_radius=config.PATCH_RADIUS,
                morse_r_onset=config.MORSE_R_ONSET, morse_r_cutoff=config.MORSE_R_CUTOFF,
                center_radius=config.CENTER_RADIUS)


@dataclass
class Frame:
    positions: np.ndarray          # (N, 2)
    orientations: np.ndarray       # (N,)
    potential_energy: float
    kinetic_energy: float
    N: int
    step: int


def _clean_attr(v):
    if v is None:
        return 'None'
    if isinstance(v, np.generic):
        return v.item()
    return v


def _decode(v):
    if isinstance(v, bytes):
        return v.decode()
    return v.item() if isinstance(v, np.generic) else v


class TrajectoryWriter:
    def __init__(self, path, attrs=None, resume=False, precision='float64',
                 chunk_frames=64):
        self.path = str(path)
        self._chunk = chunk_frames
        self.restart = None
        if resume and os.path.exists(self.path):
            self._f = h5py.File(self.path, 'a')
            self._dtype = np.dtype(self._f.attrs['precision'])
            self._truncate_to_restart()
            if attrs:
                self.set_attrs(**attrs)
        else:
            self._f = h5py.File(self.path, 'w')
            self._dtype = np.dtype(precision)
            if self._dtype not in (np.float32, np.float64):
                raise ValueError("precision must be 'float32' or 'float64'")
            self._f.attrs['schema_version'] = SCHEMA_VERSION
            self._f.attrs['software_version'] = software_version()
            self._f.attrs['precision'] = self._dtype.name
            for k, v in model_constants().items():
                self._f.attrs[k] = v
            self.set_attrs(**(attrs or {}))
            for ph in PHASES:
                g = self._f.create_group(ph)
                for name, dt in _PER_FRAME:
                    g.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dt,
                                     chunks=(self._chunk,), compression='gzip',
                                     compression_opts=4)
        self._f.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure_flat(self, g, n):
        if 'positions' in g:
            return
        rows = max(256, min(self._chunk * max(n, 1), 1 << 18))
        kw = dict(compression='gzip', compression_opts=4)
        g.create_dataset('positions', shape=(0, 2), maxshape=(None, 2),
                         dtype=self._dtype, chunks=(rows, 2), **kw)
        g.create_dataset('orientations', shape=(0,), maxshape=(None,),
                         dtype=self._dtype, chunks=(rows,), **kw)

    def set_attrs(self, **kw):
        for k, v in kw.items():
            self._f.attrs[k] = _clean_attr(v)

    def append_frame(self, positions, orientations, potential_energy, step,
                     phase='production', kinetic_energy=np.nan):
        g = self._f[phase]
        pos = np.asarray(positions).reshape(-1, 2)
        ori = np.asarray(orientations).reshape(-1)
        n = len(pos)
        if len(ori) != n:
            raise ValueError('positions and orientations disagree on N')
        self._ensure_flat(g, n)
        f = len(g['step'])
        off = len(g['positions'])
        for name, val in [('step', step), ('N', n), ('offset', off),
                          ('potential_energy', potential_energy),
                          ('kinetic_energy', kinetic_energy)]:
            g[name].resize(f + 1, axis=0)
            g[name][f] = val
        g['positions'].resize(off + n, axis=0)
        g['orientations'].resize(off + n, axis=0)
        if n:
            g['positions'][off:off + n] = pos.astype(self._dtype)
            g['orientations'][off:off + n] = ori.astype(self._dtype)

    def n_frames(self, phase):
        return len(self._f[phase]['step'])

    def _truncate(self, phase, keep):
        g = self._f[phase]
        if keep >= len(g['step']):
            return
        rows = int(g['offset'][keep])
        for name, _ in _PER_FRAME:
            g[name].resize(keep, axis=0)
        if 'positions' in g:
            g['positions'].resize(rows, axis=0)
            g['orientations'].resize(rows, axis=0)

    def _truncate_to_restart(self):
        info = _read_restart(self._f)
        self.restart = info
        if info is None:
            for ph in PHASES:
                self._truncate(ph, 0)
            return
        phase, pstep = info[0], info[1]
        if phase == 'done':
            return
        if phase == 'equilibration':
            self._truncate('production', 0)
        steps = self._f[phase]['step'][:]
        self._truncate(phase, int(np.searchsorted(steps, pstep, side='right')))

    def write_restart(self, phase, phase_step, leaves, **attrs):
        """Checkpoint: flattened MD-state arrays plus where the run stands."""
        if 'restart' in self._f:
            del self._f['restart']
        g = self._f.create_group('restart')
        g.attrs['phase'] = phase
        g.attrs['phase_step'] = int(phase_step)
        for k, v in attrs.items():
            g.attrs[k] = _clean_attr(v)
        for i, leaf in enumerate(leaves):
            g.create_dataset(f'leaf_{i}', data=np.asarray(leaf))
        self._f.flush()

    def flush(self):
        self._f.flush()

    def close(self):
        if self._f:
            self._f.flush()
            self._f.close()
            self._f = None


def _read_restart(f):
    if 'restart' not in f:
        return None
    g = f['restart']
    n = len([k for k in g if k.startswith('leaf_')])
    leaves = [g[f'leaf_{i}'][()] for i in range(n)]
    extra = {k: _decode(v) for k, v in g.attrs.items() if k not in ('phase', 'phase_step')}
    return str(g.attrs['phase']), int(g.attrs['phase_step']), leaves, extra


class TrajectoryReader:
    """Lazy reader: only the requested frames are read from disk."""

    def __init__(self, path):
        self._f = h5py.File(path, 'r')
        self.attrs = {k: _decode(v) for k, v in self._f.attrs.items()}
        self._dtype = np.dtype(self.attrs['precision'])

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def n_frames(self, phase='production'):
        return len(self._f[phase]['step'])

    def series(self, phase, name):
        """Per-frame 1D dataset: step, N, potential_energy, kinetic_energy."""
        return self._f[phase][name][:]

    def frames(self, phase='production', start=None, stop=None, step=None):
        g = self._f[phase]
        idx = range(*slice(start, stop, step).indices(len(g['step'])))
        if len(idx) == 0:
            return []
        lo, hi = min(idx), max(idx) + 1
        offs, ns = g['offset'][lo:hi], g['N'][lo:hi]
        steps, pe, ke = g['step'][lo:hi], g['potential_energy'][lo:hi], g['kinetic_energy'][lo:hi]
        # one contiguous read when the stride is small, per-frame otherwise
        bulk = idx.step == 1 and ns.sum() > 0
        if bulk:
            a, b = int(offs[0]), int(offs[-1] + ns[-1])
            P, O = g['positions'][a:b], g['orientations'][a:b]
        out = []
        for i in idx:
            j = i - lo
            n, o = int(ns[j]), int(offs[j])
            if n == 0:
                p, q = np.zeros((0, 2), self._dtype), np.zeros(0, self._dtype)
            elif bulk:
                p, q = P[o - a:o - a + n], O[o - a:o - a + n]
            else:
                p, q = g['positions'][o:o + n], g['orientations'][o:o + n]
            out.append(Frame(p, q, float(pe[j]), float(ke[j]), n, int(steps[j])))
        return out

    def read_restart(self):
        return _read_restart(self._f)


# ══════════════════════════════════════════════════════════════════════════
# Legacy loaders (pre-HDF5 outputs) behind the same reader interface
# ══════════════════════════════════════════════════════════════════════════

class LegacyReader:
    """In-memory view of one legacy trajectory with the reader interface.
    Everything is presented as the 'production' phase."""

    def __init__(self, attrs, frames):
        self.attrs = attrs
        self._frames = frames

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def n_frames(self, phase='production'):
        return len(self._frames) if phase == 'production' else 0

    def series(self, phase, name):
        if phase != 'production':
            return np.zeros(0)
        return np.array([getattr(f, name) for f in self._frames])

    def frames(self, phase='production', start=None, stop=None, step=None):
        if phase != 'production':
            return []
        return self._frames[slice(start, stop, step)]

    def read_restart(self):
        return None


def _scalar(d, k, default=None):
    if k not in d:
        return default
    return np.asarray(d[k]).reshape(-1)[0].item()


def _legacy_attrs(d, ensemble):
    attrs = dict(ensemble=ensemble, legacy=True)
    for k in ('kT', 'E_AB', 'E_AA', 'E_BB', 'rep_A', 'mu', 'seed', 'alpha_deg',
              'opening_angle', 'sample_interval'):
        v = _scalar(d, k)
        if v is not None:
            attrs[k] = v
    if 'box_size' in d:
        attrs['box_size'] = _scalar(d, 'box_size')
    elif 'box_area' in d:
        attrs['box_size'] = float(np.sqrt(_scalar(d, 'box_area')))
    return attrs


def _load_legacy_npz(path, realization=0):
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    if f'r{realization}_positions' in keys:                 # canonical cluster output
        pos = d[f'r{realization}_positions']
        ori = d[f'r{realization}_orientations']
        en = d[f'r{realization}_energy']
        n_valid = int(np.any(pos != 0, axis=(1, 2)).sum())  # unused rows are zeros
        si = int(_scalar(d, 'sample_interval', 1))
        frames = [Frame(pos[i], ori[i], float(en[i]), np.nan, pos.shape[1], i * si)
                  for i in range(n_valid)]
        return LegacyReader(_legacy_attrs(d, 'canonical'), frames)
    if 'snapshot_q' in keys:                                # GCMC run.npz, NaN padded
        q, Ns = d['snapshot_q'], d['snapshot_N']
        sweeps = d['snapshot_sweeps'] if 'snapshot_sweeps' in keys else np.arange(len(Ns))
        en = (d['E_traj'] if 'E_traj' in keys and len(d['E_traj']) == len(Ns)
              else np.full(len(Ns), np.nan))
        frames = [Frame(q[i, :int(Ns[i]), :2], q[i, :int(Ns[i]), 2], float(en[i]), np.nan,
                        int(Ns[i]), int(sweeps[i])) for i in range(len(Ns))]
        return LegacyReader(_legacy_attrs(d, 'grand_canonical'), frames)
    raise ValueError(f'unrecognised legacy npz layout: {path}')


def _load_legacy_npy(path):
    obj = np.load(path, allow_pickle=True).item()
    if 'MD_trajectories' in obj:                            # GCMC MD dump
        frames = [Frame(q[:, :2], q[:, 2], np.nan, np.nan, len(q), int(i))
                  for i, q in sorted(obj['MD_trajectories'].items())]
        return LegacyReader(dict(ensemble='grand_canonical', legacy=True), frames)
    raise ValueError(f'unrecognised legacy pickled layout: {path}')


def open_trajectory(path, realization=0):
    """Open an HDF5 trajectory or a legacy ``.npz`` / pickled ``.npy`` file."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.h5', '.hdf5'):
        return TrajectoryReader(path)
    if ext == '.npz':
        return _load_legacy_npz(path, realization)
    if ext == '.npy':
        return _load_legacy_npy(path)
    raise ValueError(f'unknown trajectory file type: {path}')
