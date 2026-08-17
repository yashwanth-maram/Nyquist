#!/usr/bin/env python3
"""
Submission self-check for KLA PS1.

Runs the required command in a clean temp directory and verifies every item on
the organisers' final checklist.

    python check_submission.py                 # uses samples/
    python check_submission.py <input-dir>     # uses your own test dir
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PASS, FAIL = 0, 0


def check(label, ok, detail=''):
    global PASS, FAIL
    mark = 'PASS' if ok else 'FAIL'
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f'[{mark}] {label}' + (f'  --  {detail}' if detail else ''))
    return ok


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'samples')

    print('=' * 68)
    print('REQUIRED FILES')
    print('=' * 68)
    check('run.py present', os.path.exists(os.path.join(HERE, 'run.py')))
    check('requirements.txt present',
          os.path.exists(os.path.join(HERE, 'requirements.txt')))
    check('README.md present', os.path.exists(os.path.join(HERE, 'README.md')))
    models = os.path.join(HERE, 'models')
    check('models/ present', os.path.isdir(models))
    if os.path.isdir(models):
        w = glob.glob(os.path.join(models, '*'))
        check('models/ contains weights', len(w) > 0,
              ', '.join(os.path.basename(x) for x in w))

    print()
    print('=' * 68)
    print('EXECUTION:  python run.py <input-dir> <output-dir>')
    print('=' * 68)
    tmp = tempfile.mkdtemp()
    out_dir = os.path.join(tmp, 'nested', 'restored')   # must be auto-created
    print(f'input : {in_dir}')
    print(f'output: {out_dir}  (does not exist yet)')
    print()

    r = subprocess.run([sys.executable, 'run.py', in_dir, out_dir],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        check('run.py exits cleanly', False, f'exit code {r.returncode}')
        shutil.rmtree(tmp, ignore_errors=True)
        summary()
        return
    check('run.py exits cleanly', True)
    check('output directory auto-created', os.path.isdir(out_dir))

    print()
    print('=' * 68)
    print('OUTPUT CONTRACT')
    print('=' * 68)
    ins = sorted(os.path.relpath(f, in_dir)
                 for f in glob.glob(os.path.join(in_dir, '**', '*.npy'),
                                    recursive=True)
                 if not os.path.basename(f).startswith('._'))
    outs = sorted(os.path.relpath(f, out_dir)
                  for f in glob.glob(os.path.join(out_dir, '**', '*.npy'),
                                     recursive=True))

    check('one output per input', len(ins) == len(outs),
          f'{len(ins)} in, {len(outs)} out')
    check('filenames match inputs exactly', ins == outs,
          'mismatch: ' + str(set(ins) ^ set(outs)) if ins != outs else '')

    bad_shape, bad_range, bad_finite, bad_res = [], [], [], []
    for n in outs:
        a = np.load(os.path.join(in_dir, n)) if n in ins else None
        b = np.load(os.path.join(out_dir, n))
        if not (b.ndim == 2 or (b.ndim == 3 and b.shape[2] == 1)):
            bad_shape.append(f'{n}{b.shape}')
        if not np.isfinite(b).all():
            bad_finite.append(n)
        elif b.min() < 0.0 or b.max() > 1.0:
            bad_range.append(f'{n}[{b.min():.3f},{b.max():.3f}]')
        if a is not None:
            ha, wa = np.squeeze(a).shape[:2]
            hb, wb = b.shape[:2]
            if (hb, wb) != (ha * 2, wa * 2):
                bad_res.append(f'{n} {ha}x{wa}->{hb}x{wb}')

    check('grayscale (H,W) or (H,W,1)', not bad_shape, ', '.join(bad_shape[:3]))
    check('no NaN / Inf', not bad_finite, ', '.join(bad_finite[:3]))
    check('values within [0,1]', not bad_range, ', '.join(bad_range[:3]))
    check('2x target resolution', not bad_res, ', '.join(bad_res[:3]))

    if outs:
        s = np.load(os.path.join(out_dir, outs[0]))
        print(f'\n       example: {outs[0]}  shape={s.shape}  dtype={s.dtype}  '
              f'range=[{s.min():.4f}, {s.max():.4f}]')

    shutil.rmtree(tmp, ignore_errors=True)
    summary()


def summary():
    print()
    print('=' * 68)
    print(f'{PASS} passed, {FAIL} failed')
    print('=' * 68)
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
