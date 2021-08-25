import argparse
import csv
import os
from collections import defaultdict

import pandas as pd

from . import underline


class HistoryTracker:
    def __init__(self, save_path=None):
        self.history = defaultdict(list)
        self.learning_rate = None
        self.save_path = save_path
        self.is_train = True

    def start_new_epoch(self, lr):
        self.history.clear()
        self.learning_rate = lr

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def step(self, metrics):
        reports = list()
        for k, v in metrics.items():
            k = k if self.is_train else f'val_{k}'
            self.history[k].append(v)
            reports.append('{} = {:.4f}'.format(k, v))

        return ', '.join(reports)

    def log(self):
        metrics = {
            k: (sum(v) / len(v) if v else 0)
            for k, v in sorted(self.history.items())
            if k.startswith('val_') != self.is_train
        }
        return ', '.join('average {} = {:.4f}'.format(name, value)
                         for name, value in metrics.items()).capitalize()

    def save(self):
        """Save averaged metrics in this epoch to csv file.
        ：保存这个epoch的平均指标到csv文件。"""

        if self.save_path is None:
            raise RuntimeError(
                'cannot save history without setting save_path.')

        keys = [k for k, _ in sorted(self.history.items())]
        metrics = [sum(v) / len(v) for _, v in sorted(self.history.items())]
        if not os.path.exists(self.save_path):
            # create a new csv file
            with open(self.save_path, 'w') as fp:
                writer = csv.writer(fp)
                writer.writerow(keys + ['lr'])
                writer.writerow(metrics + [self.learning_rate])
        else:
            with open(self.save_path, 'a') as fp:
                writer = csv.writer(fp)
                writer.writerow(metrics + [self.learning_rate])

    def report(self, last_n_epochs=5):
        """Report training history summary. 报告培训历史总结。

        Arguments:
            last_n_epochs: number of final epochs to compute average losses and metrics.
                            计算平均损失和度量的最后纪元数。
        """

        df = pd.read_csv(self.save_path)

        metrics = '\n'.join(
            f'{key:20s} {df[key][-last_n_epochs:].mean():.4f}'
            for key in df.keys() if key not in ['lr', 'loss', 'val_loss']
        )

        return underline(
            '\nTraining Summary (Avg over last 5 epochs)', style='=') + '\n' + metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', help='Path to history csv file')
    parser.add_argument('-l', '--last-n-epochs', type=int, default=5,
                        help='Number of final epochs to compute average losses and metrics')
    args = parser.parse_args()

    tracker = HistoryTracker(args.csv_path)
    tracker.report(args.last_n_epochs)
