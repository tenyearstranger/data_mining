# -*- coding = utf-8 -*-
# @Time : 2023-11-15 11:21
# @Author : mfk
# @File : test_csv.py
# @Software : PyCharm

import pymysql
import re
import datetime
import csv

if __name__ == "__main__":
    results = []

    fileIn = open('../data/traj.csv', 'r')
    results = csv.reader(fileIn, delimiter=',')

    with open('traj_lng_lat.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # 提取经度和纬度
        pattern = '\[([0-9]+.[0-9]+),([0-9]+.[0-9]+)\]'
        datas = []
        i = 0
        for result in results:
            if i == 0:
                i += 1
                data = result
                data.append("lng")
                data.append("lat")
                writer.writerow(result)
                continue
            tmp = re.match(pattern, result[4])
            lng = tmp.group(1)
            lat = tmp.group(2)
            data = []
            for row in result:
                data.append(row)
            data.append(lng)
            data.append(lat)
            writer.writerow(data)
        print("data buildOver")
    fileIn.close()
