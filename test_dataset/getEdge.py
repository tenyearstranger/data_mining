import re
import csv
import pandas as pd
def get_lng_lat(lng_lat_string):
    # 提取经度和纬度
    pattern = ',?\[(\d+\.\d+),(\d+\.\d+)\]'
    tmp = re.match(pattern, lng_lat_string)
    lng = float(tmp.group(1))
    lat = float(tmp.group(2))
    return lng, lat

if __name__ == "__main__":
    pattern = re.compile(u'\[\d+\.\d+,\d+\.\d+\]')
    dataIns = pd.read_csv('../data/road.csv')
    dataOuts = []
    for dataIn in dataIns.iloc:
        s = str(dataIn[1])
        s = s.replace(" ","")
        groups = pattern.findall(s)
        lngs = []
        lats = []
        for group in groups :
            lng,lat = get_lng_lat(group)
            lngs.append(lng)
            lats.append(lat)
        for i in range(1,len(lngs)):
            dataOut = [lngs[i-1],lats[i-1],lngs[i],lats[i]]
            dataOuts.append(dataOut)

    with open('../extendedData/edge.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(dataOuts)