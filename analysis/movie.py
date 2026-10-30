"""
Trajectory GIF: particle cores with distinguishable A/B patches, optionally
coloured by cluster with the colour map shared with the snapshot and histogram
plots (:mod:`analysis.structure_plots`).

``make_gif(source, out_path, start, stop, stride, fps, figsize, ...)`` writes
one GIF frame per stored frame in ``range(start, stop, stride)``.  With
``unwrap=True`` every cluster is drawn contiguous (a ring crossing the periodic
boundary is not torn apart) and the box outline is dotted; otherwise particles
are wrapped into the box and drawn with periodic images at the edges.  Needs
Pillow (``pip install pillow``); a missing install raises ``ImportError`` with
that instruction.  Headless-safe (Agg canvas, no ``pyplot``).

Command line::

    python -m analysis.movie FILE OUT.gif [--start 0] [--stop N] [--stride 1]
        [--fps 10] [--figsize 5] [--no_clusters] [--unwrap] [--phase production]
"""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from analysis.bond_graph import classify_frame
from analysis.structure_plots import _open, draw_frame


def _pillow():
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('GIF output needs Pillow: pip install pillow') from e
    return Image


def make_gif(source, out_path, start=0, stop=None, stride=1, fps=10, figsize=5.0,
             phase='production', color_clusters=True, unwrap=False,
             bond_distance=None, dpi=80):
    """Write a GIF of the trajectory; returns the number of GIF frames."""
    Image = _pillow()
    owned = isinstance(source, (str, Path))
    reader = _open(source)
    try:
        L, theta = float(reader.attrs['box_size']), float(reader.attrs['opening_angle'])
        frames = reader.frames(phase, start, stop, stride)
    finally:
        if owned:
            reader.close()
    if not frames:
        raise ValueError(f'no {phase} frames in range({start}, {stop}, {stride})')
    images = []
    for k, fr in enumerate(frames):
        fig = Figure(figsize=(figsize, figsize), dpi=dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.01, 0.01, 0.98, 0.93])
        s = classify_frame(fr.positions, fr.orientations, L, theta, bond_distance) \
            if (color_clusters or unwrap) and fr.N else None
        draw_frame(ax, fr.positions, fr.orientations, L, theta, s,
                   color_clusters=color_clusters, unwrap=unwrap)
        ax.set_title(f'{phase}  step {fr.step}  N = {fr.N}', fontsize=9)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3]))
    images[0].save(out_path, save_all=True, append_images=images[1:],
                   duration=int(round(1000 / fps)), loop=0)
    return len(images)


def main(argv=None):
    p = argparse.ArgumentParser(description='Trajectory GIF')
    p.add_argument('file')
    p.add_argument('out')
    p.add_argument('--phase', default='production')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--stop', type=int, default=None)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--figsize', type=float, default=5.0)
    p.add_argument('--no_clusters', action='store_true')
    p.add_argument('--unwrap', action='store_true')
    a = p.parse_args(argv)
    n = make_gif(a.file, a.out, a.start, a.stop, a.stride, a.fps, a.figsize, a.phase,
                 not a.no_clusters, a.unwrap)
    print(f'Wrote {a.out} ({n} frames)')


if __name__ == '__main__':
    main()
