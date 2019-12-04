import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sb
import textblob as tb

file_name = "../data/pizza/pizza.json"

data = pd.read_json(file_name)
cols = data.columns

"""
Drop irrelevant columns
"""

keep_cols = [0, 1, 2, 5, 6, 8, 22, 25, 28, 29]
data = data[cols[keep_cols]]

"""
Empty class
"""


class Analysis(object):
    pass


"""
Group data, sentiment analysis, and analysis summary into Analysis object
"""


def stats(data):
    res = Analysis()

    sent_text = [tb.TextBlob(s).sentiment.polarity for s in data[data.columns[4]]]
    sent_head = [tb.TextBlob(s).sentiment.polarity for s in data[data.columns[5]]]

    net_votes = data[data.columns[2]] - data[data.columns[1]]

    data = data.assign(
        sentiment_text=sent_text, sentiment_head=sent_head, net_votes=net_votes
    )
    desc = [data[c].describe() for c in data.columns]

    res.cols = data.columns
    res.data = data
    res.corr = data.corr()
    res.desc = desc

    return res


"""
These requests are fulfilled
"""

is_settled = data[data.columns[6]]
settled = data[is_settled]

"""
These requests are unfulfilled
"""

is_lacking = np.invert(is_settled)
lacking = data[is_lacking]

"""
stats_s = stats(settled)
stats_l = stats(lacking.sample(len(settled)))
"""

stats_s = stats(settled)
stats_l = stats(lacking)

"""
Graph ticks
"""

x_ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
y_ticks = [0, 1, 2, 3]

"""
Density plot
"""

g = sb.kdeplot(stats_s.data[stats_s.cols[10]])
r = g.set(xticks=x_ticks, yticks=y_ticks)

"""
Density plot
"""

g = sb.kdeplot(stats_l.data[stats_l.cols[10]])
r = g.set(xticks=x_ticks, yticks=y_ticks)

stats_s.desc[10]
stats_l.desc[10]

"""
Abstraction for finding statistical significance
"""


def ttest(a, b):
    return sp.stats.ttest_ind(a, b).pvalue


def ttest_data(a, b):
    keep_cols = [1, 2, 3, 6, 7, 10, 11, 12]
    cols = len(keep_cols)

    a = a[a.columns[keep_cols]]
    b = b[b.columns[keep_cols]]

    pvals = [ttest(a[c], b[c]) for c in a.columns]
    res = {a.columns[i]: pvals[i] for i in range(cols)}
    return res


cmp = ttest_data(stats_s.data, stats_l.data)

"""
Filter function
"""


def is_sig(v):
    return v < 0.05


sig = {c: v for (c, v) in cmp.items() if is_sig(v)}

"""
Significant differences
"""

sig
