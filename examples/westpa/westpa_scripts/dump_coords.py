#!/usr/bin/env python
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='dump npy')
parser.add_argument('--zero', default=False, action='store_true',
                help='dump zero\'th step')
parser.add_argument('--output', metavar='textfile', default='/dev/stdout',
                help='output text file')
parser.add_argument('input', metavar='npyfile',
                help='input npy file')

args = parser.parse_args()


traj = np.load(args.input)
if traj.ndim == 1:
    np.savetxt(args.output, traj[np.newaxis, :])
elif traj.ndim == 2:
    if not args.zero:
        traj = traj[1:, :]
    np.savetxt(args.output, traj)
else:
    raise RuntimeError('bad ndim')