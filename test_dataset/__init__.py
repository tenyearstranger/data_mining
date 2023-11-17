# -*- coding = utf-8 -*-
# @Time : 2023-11-14 16:03
# @Author : mfk
# @File : __init__.py.py
# @Software : PyCharm
import datetime
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
searchTraj = 'select `entity_id`,`time`,`coordinates` from `traj` where `traj_id` = %s;'


def get_lng_lat(lng_lat_string):
    # 提取经度和纬度
    pattern = '\[([0-9]+.[0-9]+),([0-9]+.[0-9]+)\]'
    tmp = re.match(pattern, lng_lat_string)
    lng = float(tmp.group(1))
    lat = float(tmp.group(2))
    return lng, lat


def getTraj(trajId):
    times = []
    lngs = []
    lats = []
    cursor.execute(searchTraj, trajId)
    results = cursor.fetchall()
    for result in results:
        lng, lat = get_lng_lat(result[2])
        lngs.append(lng)
        lats.append(lat)
        time = datetime.datetime.strptime(result[1], '%Y-%m-%dT%H:%M:%SZ')
        times.append(time)
    return times, lngs, lats


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

    selectEntityId = 'select `entity_id` from `traj` where `traj_id` = %s'
    for i in range(min_id, min_id + 5):
        cursor.execute(selectEntityId, i)
        entity_id = cursor.fetchone()
        print("extend " + str(i) + "--------------")
        times, lngs, lats = getTraj(i)
        # 将时间转化为浮点数
        plots = []
        timeFloats = []
        for time in times:
            timeFloats.append(time.timestamp())
        for j in range(1, len(timeFloats)):
            plot = (timeFloats[j-1] + timeFloats[j])/2
            plots.append(plot)
        # 进行插值
        res1 = np.interp(plots, timeFloats, lngs)
        res2 = np.interp(plots, timeFloats, lats)
        print(res1)
        print(res2)
