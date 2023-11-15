# -*- coding = utf-8 -*-
# @Time : 2023-11-14 16:03
# @Author : mfk
# @File : __init__.py.py
# @Software : PyCharm

import pymysql
import re
import datetime

if __name__ == "__main__":
    conn = pymysql.connect(
        host='127.0.0.1',  # 主机名
        port=3306,  # 端口号，MySQL默认为3306
        user='root',  # 用户名
        password='bh030618mfk',  # 密码
        database='dm_hw',  # 数据库名称
    )

    cursor = conn.cursor()
    cursor.execute("select * from traj where holidays = 1;")
    results = cursor.fetchall()
    # print(len(results))

    # for result in results:
    #     time = datetime.datetime.strptime(result[1], "%Y-%m-%dT%H:%M:%SZ");
    #     print(time.year)
    #     print(time.month)
    #     print(time.day)

    # 提取经度和纬度
    pattern = '\[([0-9]+.[0-9]+),([0-9]+.[0-9]+)\]'
    updateSql = 'UPDATE `traj_lng_lat` SET `lng` = %s, `lat` = %s WHERE `id` = %s'
    datas = []
    for result in results:
        tmp = re.match(pattern, result[4])
        lng = tmp.group(1)
        lat = tmp.group(2)
        data = (lng, lat, result[0])
        print(data)
        datas.append(data)
    print("data buildOver")
    print(datas)
    print("--------------------------------------------------")
    cursor.execute(updateSql,datas[0])
    conn.commit()
    print(datas[0])
    cursor.executemany(updateSql, datas)
    conn.commit()
