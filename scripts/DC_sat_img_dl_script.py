import numpy as np
import censusgeocode as cg
import img_dl
import geoid_income_utils
import DC_geoid_income_utils

SaveLoc = r"../data/DC_1/Images"

GEO_ID_income_dict = DC_geoid_income_utils.readFromFile()
index = 1000000
iterations = 0
labels = {}
locations = {}

for lat in np.arange(38.803611, 38.997022, 0.002):
    for lon in np.arange(-77.129420, -76.900714, 0.002):
        lat_rd = round(lat, 6)
        lon_rd = round(lon, 6)     

        cg_failed = False
        for i in range(5):
            try:
                cg_obj = cg.coordinates(lon_rd, lat_rd)
                cg_failed = False
                break
            except:
                cg_failed = True
                pass
        if cg_failed:
            continue

        raw_geoid = cg_obj['2010 Census Blocks'][0]['GEOID']
        geoid = raw_geoid[0:11]

        loc = str(lat_rd) + "," + str(lon_rd)
        
        if geoid in GEO_ID_income_dict: #GEO_ID is in database and streetview image exists
            #Get image
            try:
                img_dl.getSatelliteImage(loc, SaveLoc, index)
            except:
                continue

            #Store labels
            income = GEO_ID_income_dict[geoid]
            labels[index] = income

            #Get image location
            locations[index] = loc

            if index % 100 == 0:
                print("Downloaded " + str(index - 1000000 + 1) + " images")
            index += 1

        if iterations % 100 == 0:
            print("Iteration " + str(iterations) + " complete!")
            geoid_income_utils.writeLabelsToFile("DC_1_index_to_latlong", locations)
            geoid_income_utils.writeLabelsToFile("DC_1_labels", labels)
        iterations += 1
        # print(str(iterations))

geoid_income_utils.writeLabelsToFile("DC_1_index_to_latlong", locations)
geoid_income_utils.writeLabelsToFile("DC_1_labels", labels)
            