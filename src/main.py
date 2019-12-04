import numpy as np
import pandas as pd
import seaborn as sb
import textblob as tb

file_name = "../data/pizza/pizza.json"

data = pd.read_json(file_name)
cols = data.columns

keep_cols = [0, 1, 2, 5, 6, 8, 22, 25, 28, 29]
data = data[cols[keep_cols]]


class Analysis(object):
    pass


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


is_settled = data[data.columns[6]]
settled = data[is_settled]

is_lacking = np.invert(is_settled)
lacking = data[is_lacking]

stats_s = stats(settled)
stats_l = stats(lacking.sample(len(settled)))

x_ticks = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
y_ticks = [0, 1, 2, 3]

g = sb.kdeplot(stats_s.data[stats_s.cols[10]])
r = g.set(xticks=x_ticks, yticks=y_ticks)

g = sb.kdeplot(stats_l.data[stats_l.cols[10]])
r = g.set(xticks=x_ticks, yticks=y_ticks)

stats_s.desc[10]
stats_l.desc[10]
