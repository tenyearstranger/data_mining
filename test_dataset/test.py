# -*- coding = utf-8 -*-
# @Time : 2023-11-15 10:43
# @Author : mfk
# @File : test.py
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
    lng = 3
    lat = 3
    id = 1
    updateSql = 'UPDATE `traj_lng_lat` SET `lng` = %s, `lat` = %s WHERE `id` = %s'
    data = (2, 2, 0)
    cursor.execute(updateSql, data)
    conn.commit()
