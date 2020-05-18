import numpy as np
import censusgeocode as cg
import img_dl
import geoid_income_utils

SaveLoc = r"./StreetView Images"

GEO_ID_income_dict = geoid_income_utils.readFromFile()
index = 1000000
iterations = 0
labels = {}
locations = {}
# print("start looping")
for lat in np.arange(33.0672, 34.0854, 0.01):
    for lon in np.arange(-112.7143, -111.4015, 0.01):
        #TODO: get GEO_ID
        # print(lon)
        # print(lat)
        lat_rd = round(lat, 4)
        lon_rd = round(lon, 4)
        cg_obj = cg.coordinates(lon_rd, lat_rd)
        raw_geoid = cg_obj['2010 Census Blocks'][0]['GEOID']
        geoid = raw_geoid[0:11]
        # print(geoid)
        # print(GEO_ID_income_dict)

        # print("first check")
        loc = str(lat_rd) + "," + str(lon_rd)
        if geoid in GEO_ID_income_dict and img_dl.checkLatLongImage(loc) == "OK": #GEO_ID is in database and streetview image exists
            #Store labels
            income = GEO_ID_income_dict[geoid]
            labels[index] = income

        
            #Get image and store location
            locations[index] = loc
            # print(loc)
            img_dl.getStreetViewImage(loc, SaveLoc, index);

            if index % 100 == 0:
                print("Downloaded " + str(index - 1000000 + 1) + " images")
            index += 1

        if iterations % 100 == 0:
            print("Iteration " + str(iterations) + " complete!")
        iterations += 1

geoid_income_utils.writeLabelsToFile("index_to_latlong", locations)
geoid_income_utils.writeLabelsToFile("labels", labels)
            