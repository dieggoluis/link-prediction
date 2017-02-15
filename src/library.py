import csv

def read_files(file_names):
    """ read data and return a list of files """
    assert type(file_names) == list
    
    file_list = []
    for file_name in file_names:
        print "Reading ", file_name
        with open(file_name, "r") as f:
            reader = csv.reader(f)
            file_list.append(list(reader))

    return file_list
