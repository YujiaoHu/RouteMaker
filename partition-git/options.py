import argparse
import os
import sys

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="model parameters")
    parser.add_argument('--cuda', type=int, default=1, help='which cuda')
    parser.add_argument('--anum', type=int, default=10, help="agent num")
    parser.add_argument('--cnum', type=int, default=100, help="city num, including depot")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--trainIns', type=int, default=16, help='S-sample, S setting')
    parser.add_argument("--modelpath", default=os.path.join(os.getcwd(), "../savemodel"))
    parser.add_argument("--entropy_coeff", type=float, default=0.2, help="entropy coeff")
    parser.add_argument("--obj", type=str, default='minmax', help="minmax or minsum")
    parser.add_argument("--problem", type=str, default='cfa', help="cfa or cfd or sdmtsp")
    
    ######  eval parameters #####
    parser.add_argument("--decode_strategy", type=str, default='improvement', help="greedy or sampling or improvement")
    parser.add_argument("--sampling_number", type=int, default=10, help="sampling number using smapling decoding")
    parser.add_argument("--moveTL", type=float, default=2.0, help="improvement time usage")
    
    opts = parser.parse_args(args)
    return opts
