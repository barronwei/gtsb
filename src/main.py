import pandas
from textblob import TextBlob as tb

file_name = "../data/pizza/pizza.json"

data = pandas.read_json(file_name)
drop_cols = [1, 2, 4, 5, 7, 10, 11, 14, 15]

data.drop(data.columns[drop_cols], axis=1, inplace=True)


class Analysis(object):
    pass


def stats(data):
    res = Analysis()

    sent_text = [tb(s).sentiment.polarity for s in data[data.columns[1]]]
    sent_head = [tb(s).sentiment.polarity for s in data[data.columns[2]]]

    data = data.assign(sentiment_text=sent_text, sentiment_head=sent_head)
    desc = [data[c].describe() for c in data.columns]

    res.corr = data.corr()
    res.desc = desc

    return res


is_settled = data[data.columns[0]] != "N/A"
settled = data[is_fulfilled]

is_lacking = data[data.columns[0]] == "N/A"
lacking = data[is_lacking]

stats_settled = stats(settled)
stats_lacking = stats(lacking)
