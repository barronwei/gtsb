import pandas

file_name = "../data/pizza/pizza.json"

data = pandas.read_json(file_name)
drop_cols = [1, 2, 4, 5, 7, 10, 11, 14, 15]

data.drop(data.columns[drop_cols], axis=1, inplace=True)

is_fulfilled = data[data.columns[0]] != "N/A"
complete = data[is_fulfilled]

len(complete)

complete.head()
