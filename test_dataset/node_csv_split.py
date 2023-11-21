import re
import csv

data_to_write = []
with open('../data/node.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)

    with open('node_lng_lat.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        datas = []
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                data = row
                data.append("lng")
                data.append("lat")
                writer.writerow(row)
                continue
            coordinates = row[2].replace("[", "").replace("]", "").split(",")
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
            print(f"经度: {longitude}, 纬度: {latitude}")
            data = []
            for result in row:
                data.append(result)
            data.append(longitude)
            data.append(latitude)
            writer.writerow(data)
        print("data buildOver")