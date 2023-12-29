#get all values from a dataframe into a list
def listify(x):
    x = x.values.tolist()
    return [item for row in x for item in row]
