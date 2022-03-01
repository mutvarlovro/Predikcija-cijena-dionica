import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

matplotlib.rcParams["figure.figsize"] = (8, 6)

class plot_context:
    def __init__(self, **kwargs):
        self.params = defaultdict(lambda: None)
        # self.params.update(standalone=True, grid=False)
        self.params.update(kwargs)

    def __getattr__(self, item):
        return self.params.get(item, None)

    def __enter__(self):
        if self.standalone or self.figsize: plt.figure(figsize=self.figsize)
        if self.subplot: plt.subplot(*self.subplot) if isinstance(self.subplot, tuple) else plt.subplot(self.subplot)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.title is not None: plt.title(self.title)
        if self.suptitle is not None: plt.suptitle(self.suptitle)
        if self.legend is not None: plt.legend(self.legend, loc="best")
        if self.xlabel is not None: plt.xlabel(self.xlabel)
        if self.ylabel is not None: plt.ylabel(self.ylabel)
        if self.xscale is not None: plt.xscale(self.xscale)
        if self.yscale is not None: plt.yscale(self.yscale)
        if self.xticks is not None: plt.xticks(self.xticks, self.xlabels, rotation=self.xticksrotation)
        if self.yticks is not None: plt.yticks(self.yticks, self.ylabels, rotation=self.yticksrotation)
        if self.xlim is not None: plt.xlim(*self.xlim)
        if self.ylim is not None: plt.ylim(*self.ylim)
        if self.colorbar: plt.colorbar()
        if self.grid: plt.grid()
        if self.tight: plt.tight_layout()
        if self.export: plt.savefig(self.export, bbox_inches="tight")
        if self.standalone or self.show: plt.show()
        if self.standalone or self.show: plt.close()

def histogram(data, **kwargs):
    with plot_context(**kwargs):
        plt.hist(data)

def adv_histogram(preds, targets, bins=10, **kwargs):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    delta = 1.0 / bins
    count_1 = np.sum(targets)

    def gen_bin_ratios():
        for left, right in zip(bin_edges, bin_edges[1:]):
            cond = (preds >= left) & (preds < right)
            bin_count = np.sum(cond)
            bin_count_1 = np.sum(targets[cond])
            yield bin_count_1 / count_1, bin_count

    bin_ratios, counts = zip(*list(gen_bin_ratios()))
    counts = counts / np.sum(counts)
    # bin_ratios = np.log2(bin_ratios)
    bin_ratios = np.array(bin_ratios)
    min_ratio, max_ratio = bin_ratios.min(), bin_ratios.max()
    bin_ratios_scaled = (bin_ratios - min_ratio) / (max_ratio - min_ratio)
    colormap = plt.get_cmap("viridis")

    with plot_context(colorbar=False, **kwargs):
        plt.bar(bin_edges[:-1] + delta / 2, counts, delta, color=colormap(bin_ratios_scaled))
        plt.xticks(np.arange(0.0, 1.01, 0.1))
        for i, (bin_ratio, count) in enumerate(zip(bin_ratios, counts)):
            plt.text(0.1 * i + 0.05, count + 0.005 * max_ratio, f"{bin_ratio:7.3%}", horizontalalignment="center")

def plot(x1, x2=None, c="-", **kwargs):
    with plot_context(**kwargs):
        plt.plot(x1, x2, c)
