import json
import pickle as pkl


def save_json(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def load_json(filename):
    data = {}
    with open(filename, "r") as json_data:
        data = json.load(json_data)

    return data


def print_json(filename=None, data=None):
    data_json = data
    if filename is not None:
        data_json = load_json(filename)

    if data_json is not None:
        print(json.dumps(data_json, sort_keys=True,
                         indent=4, separators=(',', ': ')))


def save(filename, data):
    with open(filename, 'wb') as file:
        pkl.dump(data, file)


def load(filename):
    try:
        data = pkl.load(open(filename, 'rb'))
    except OSError:
        data = []

    return data


def save_list(file, some_list):
    with open(file, "w") as fobj:
        for element in some_list:
            fobj.write("%s\n" % element)
