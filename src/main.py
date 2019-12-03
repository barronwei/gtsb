import pandas
from textblob import TextBlob as tb

file_name = "../data/pizza/pizza.json"

data = pandas.read_json(file_name)
drop_cols = [1, 2, 4, 5, 7, 10, 11, 14, 15]

data.drop(data.columns[drop_cols], axis=1, inplace=True)

is_fulfilled = data[data.columns[0]] != "N/A"
complete = data[is_fulfilled]
cols = complete.columns

len(complete)
complete.head()

sent_text = [tb(s).sentiment.polarity for s in complete[complete_cols[1]]]
sent_head = [tb(s).sentiment.polarity for s in complete[complete_cols[2]]]
comp = complete.assign(sentiment_text=sent_text, sentiment_head=sent_head)

comp_cols = comp.columns
stats = [comp[c].describe() for c in comp_cols]

comp.corr()
