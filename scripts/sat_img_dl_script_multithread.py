import numpy as np
import censusgeocode as cg
import img_dl
import geoid_income_utils
from concurrent.futures import ThreadPoolExecutor
import threading

SaveLoc = r"../data/AZ_4/Images"

GEO_ID_income_dict = geoid_income_utils.readFromFile()
index = 4000000
iterations = 0
labels = {}
locations = {}

def latlon_dl(lat, lon):
    lat_rd = round(lat, 6)
    lon_rd = round(lon, 6)     

    cg_failed = False
    for i in range(5):
        try:
            cg_obj = cg.coordinates(lon_rd, lat_rd)
            cg_failed = False
            return
        except:
            cg_failed = True
            pass
    if cg_failed:
        return

    raw_geoid = cg_obj['2010 Census Blocks'][0]['GEOID']
    geoid = raw_geoid[0:11]

    loc = str(lat_rd) + "," + str(lon_rd)

    lock.acquire()
    if geoid in GEO_ID_income_dict: #GEO_ID is in database and streetview image exists
        #Get image
        try:
            img_dl.getSatelliteImage(loc, SaveLoc, index)
        except:
            return

        #Store labels
        income = GEO_ID_income_dict[geoid]
        labels[index] = income

        #Get image location
        locations[index] = loc

        if index % 100 == 0:
            print("Downloaded " + str(index - 4000000 + 1) + " images")
        index += 1

    if iterations % 100 == 0:
        print("Iteration " + str(iterations) + " complete!")
        geoid_income_utils.writeLabelsToFile("AZ_4_index_to_latlong", locations)
        geoid_income_utils.writeLabelsToFile("AZ_4_labels", labels)
    iterations += 1
    lock.release()
    


with ThreadPoolExecutor(max_workers=32) as executor:
    for lat in np.arange(33.207042, 33.703049, 0.002):
        for lon in np.arange(-112.469886, -111.584597, 0.002):
            executor.submit(latlon_dl, lat, lon)

geoid_income_utils.writeLabelsToFile("AZ_4_index_to_latlong", locations)
geoid_income_utils.writeLabelsToFile("AZ_4_labels", labels)
            
