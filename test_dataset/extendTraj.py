# -*- coding = utf-8 -*-
# @Time : 2023-11-21 16:33
# @Author : mfk
# @File : extendTraj.py
# @Software : PyCharm
import csv
import datetime
import re
import pandas as pd
import numpy as np

timeFormat = '%Y-%m-%dT%H:%M:%SZ'
dataIns = pd.read_csv('../data/traj.csv')
dataInCnt = 0


#

def get_lng_lat(lng_lat_string):
    # 提取经度和纬度
    pattern = '\[([0-9]+.[0-9]+),([0-9]+.[0-9]+)\]'
    tmp = re.match(pattern, lng_lat_string)
    lng = float(tmp.group(1))
    lat = float(tmp.group(2))
    return lng, lat


def getTraj(trajId):
    global dataInCnt
    times = []
    lngs = []
    lats = []
    dataIn = dataIns.iloc[dataInCnt]
    while dataIn[3] == trajId:
        lng, lat = get_lng_lat(dataIn[4])
        lngs.append(lng)
        lats.append(lat)
        time = datetime.datetime.strptime(dataIn[1], timeFormat)
        times.append(time)
        dataInCnt += 1
        dataIn = dataIns.iloc[dataInCnt]
    return times, lngs, lats


def extendTraj():
    global dataInCnt
    # 定义traj_id的最大值和最小值
    min_id = 0
    max_id = 770

    datas = []  # 最终的数据容器
    for i in range(min_id, max_id):
        print("%d--------------------"%i)
        dataIn = dataIns.iloc[dataInCnt]
        if dataIn[3] != i:
            continue
        entity_id = dataIn[2]
        times, lngs, lats = getTraj(i)
        # 将时间转化为浮点数
        plots = []  # 插值点
        timeFloats = []
        for time in times:
            timeFloats.append(time.timestamp())
        for j in range(1, len(timeFloats)):
            timeInterp = (timeFloats[j - 1] + timeFloats[j]) / 2
            plots.append(timeInterp)
        # 进行插值
        res1 = np.interp(plots, timeFloats, lngs)
        res2 = np.interp(plots, timeFloats, lats)
        # 构造写入数据
        datas.append([entity_id, i, times[0].strftime(timeFormat), lngs[0], lats[0]])
        for j in range(1, len(timeFloats)):
            timeString = datetime.datetime.fromtimestamp(timeFloats[j - 1]).strftime(timeFormat)
            datas.append([entity_id, i, timeString, res1[j - 1], res2[j - 1]])
            datas.append([entity_id, i, times[j].strftime(timeFormat), lngs[j], lats[j]])
        # for data in datas:
        #     print(data)
    # 写入csv
    with open('../extendedData/extendedTraj.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(datas)


if __name__ == "__main__":
    extendTraj()
