# -*- coding = utf-8 -*-
# @Time : 2023-11-15 10:02
# @Author : mfk
# @File : __init__.py.py
# @Software : PyCharm


import transbigdata as tbd
import pandas as pd
# 地图匹配包
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz

if __name__ == "__main__":
    data = pd.read_csv('../data/road.csv', header=None)
    print(data.columns)
    data.columns = ['id', 'coordinates', 'highway', 'length', 'lanes',
                    'tunnel', 'bridge', 'maxspeed', 'width', 'alley', 'roundabout']
    # col = ['Vehicleid', 'Time', 'Lng', 'Lat', 'OpenStatus']
    # oddata = tbd.taxigps_to_od(data, col=col)
    # data_deliver, data_idle = tbd.taxigps_traj_point(data, oddata, col=col)
