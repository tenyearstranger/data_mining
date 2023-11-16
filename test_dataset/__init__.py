# -*- coding = utf-8 -*-
# @Time : 2023-11-14 16:03
# @Author : mfk
# @File : __init__.py.py
# @Software : PyCharm

import re

import numpy as np
import pymysql

conn = pymysql.connect(
    host='127.0.0.1',  # 主机名
    port=3306,  # 端口号，MySQL默认为3306
    user='root',  # 用户名
    password='bh030618mfk',  # 密码
    database='dm_hw',  # 数据库名称
)
cursor = conn.cursor()
searchTraj = 'select `entity_id`,`coordinates` from `traj` where `traj_id` = %s;'


def get_lng_lat(lng_lat_string):
    # 提取经度和纬度
    pattern = '\[([0-9]+.[0-9]+),([0-9]+.[0-9]+)\]'
    tmp = re.match(pattern, lng_lat_string)
    lng = float(tmp.group(1))
    lat = float(tmp.group(2))
    return lng, lat


def getTraj(trajId):
    lngs = []
    lats = []
    cursor.execute(searchTraj, trajId)
    results = cursor.fetchall()
    for result in results:
        lng, lat = get_lng_lat(result[1])
        lngs.append(lng)
        lats.append(lat)
    return lngs, lats


if __name__ == "__main__":
    # 找到traj_id的最大值和最小值
    tmpSearch = 'select max(traj_id) from traj;'
    cursor.execute(tmpSearch)
    max_id = cursor.fetchone()
    max_id = int(max_id[0])
    tmpSearch = 'select min(traj_id) from traj'
    cursor.execute(tmpSearch)
    min_id = cursor.fetchone()
    min_id = int(min_id[0])

    for i in range(min_id, min_id + 5):
        print("extend " + str(i) + "--------------")
        datas = getTraj(i)
        lngs, lats = getTraj(i)
        lngPlot = (lngs[0] + lngs[1]) / 2
        res = np.interp([lngPlot], lngs, lats)
        print(res)
