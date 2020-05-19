import numpy as np
import censusgeocode as cg
import img_dl
import geoid_income_utils

SaveLoc = r"./StreetView Images"
SaveLoc2 = r"./Batch 2/StreetViewImages"

GEO_ID_income_dict = geoid_income_utils.readFromFile()
index = 2000000
iterations = 0
labels = {}
locations = {}

for lat in np.arange(33.648311, 33.860684, 0.002):
    for lon in np.arange(-84.465924, -84.349881, 0.002):
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
        img_exists = False
        try:
            img_exists = img_dl.checkLatLongImage(loc) == "OK"
        except:
            continue
        if geoid in GEO_ID_income_dict and img_exists: #GEO_ID is in database and streetview image exists
            #Get image
            try:
                img_dl.get360StreetViewImage(loc, SaveLoc2, index)
            except:
                continue

            #Store labels
            income = GEO_ID_income_dict[geoid]
            labels[index] = income

            #Get image location
            locations[index] = loc

            if index % 100 == 0:
                print("Downloaded " + str(index - 2000000 + 1) + " images")
            index += 1

        if iterations % 100 == 0:
            print("Iteration " + str(iterations) + " complete!")
            geoid_income_utils.writeLabelsToFile("Atlanta_index_to_latlong", locations)
            geoid_income_utils.writeLabelsToFile("Atlanta_labels", labels)
        iterations += 1
        # print(str(iterations))

geoid_income_utils.writeLabelsToFile("Atlanta_index_to_latlong", locations)
geoid_income_utils.writeLabelsToFile("Atlanta_labels", labels)
            