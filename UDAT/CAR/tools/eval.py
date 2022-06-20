from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import functools
# sys.path.append("./")

import os
import argparse
import sys

# sys.path.append(os.getcwd()+'/toolkit')
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import UAVDataset, NAT_LDataset, UAVDARK70Dataset, NATDataset
from toolkit.evaluation import OPEBenchmark


parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str, default='./results', # './results', './hp_search_results/checkpoint_e17'
                    help='tracker result path')
parser.add_argument('--dataset', '-d', type=str, default='NAT',  # UAVDark70, NAT_L
                    help='dataset name')
parser.add_argument('--num', '-n', default=10, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level', default=True,
                    action='store_true')
parser.set_defaults(show_video_level=False)
args = parser.parse_args()


def main():
    tracker_dir = os.path.join(args.tracker_path, args.dataset)
    trackers = glob(os.path.join(args.tracker_path,
                                 args.dataset,
                                 args.tracker_prefix+'*'))
    trackers = [x.split('/')[-1] for x in trackers]

    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    # root = os.path.realpath(os.path.join(os.path.dirname(__file__),
    #                         '../testing_dataset'))
    root = '/YOUR/PATH/TO/dataset'
    # root = os.path.join(root, args.dataset)
    if 'UAVDark70' in args.dataset:
        dataset = UAVDARK70Dataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
    elif 'NAT' == args.dataset:
        dataset = NATDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
    elif 'NAT_L' in args.dataset:
        dataset = NAT_LDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)


if __name__ == '__main__':
    main()
