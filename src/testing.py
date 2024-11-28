import pickle
import os
import sys
import csv
from datetime import datetime

objPath = sys.argv[1]

def write_csv_file(path, head, data):
    if not os.path.exists('resultScenarios'):
        os.mkdir('resultScenarios')

    paths = os.path.join('resultScenarios', str(path) + '.csv')

    try:
        with open(paths, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')
            if head is not None:
                writer.writerow(head)
            for row in data:
                writer.writerow(row)
            print("Write a CSV file to path %s Successful." % paths)
    except Exception as e:
        print("Write a CSV file to path: %s, Case: %s" % (path, e))

if os.path.isfile(objPath):
    with open(objPath, "rb") as f_f:
        result = pickle.load(f_f)

    print(result)

    # now = datetime.now()
    # date_time = now.strftime("%m-%d-%Y-%H-%M-%S")

    # storage = []
    # for pop in result:
    #     print("ttc, smoothness, collision", pop.ttc, pop.smoothness, pop.isCollision)
    #     storage.append([pop.ttc, pop.smoothness, pop.isCollision])

    # write_csv_file("Populations_information_" + date_time, ["ttc", "smoothness", "collision"], storage)
