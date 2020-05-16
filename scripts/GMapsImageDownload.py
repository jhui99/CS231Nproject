import urllib.parse
import urllib.request, os
import requests
import LatLongData

myloc = r"./StreetView Images" #Replace with correct location
mysatloc = r"./Location Satellite Images"
key = "&key=" + "AIzaSyA8RcVdojD1-GWoNTOPSES8KPGC1swbp2w" #got banned after ~100 requests with no key or after 25,000 requests without signature

def getStreetViewImage(Loc, SaveLoc, index):
    base = "https://maps.googleapis.com/maps/api/streetview?size=224x224&source=outdoor&location="
    MyUrl = base + urllib.parse.quote_plus(Loc) + key #added url encoding
    # MyUrl = GMapsCredentials.sign_url(MyUrl)
    #print(MyUrl)
    fi = str(index) + ".jpg"
    urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc, fi))

def checkLatLongImage(Loc):
    metadata_base = "https://maps.googleapis.com/maps/api/streetview/metadata?&location=" 
    metadata_url = metadata_base + urllib.parse.quote_plus(Loc) + key #added url encoding
    response = requests.post(metadata_url)
    response_data = response.json()
    return response_data["status"]

def getSatelliteImage(Loc, SaveLoc, index):
    base = "https://maps.googleapis.com/maps/api/staticmap?size=640x640&maptype=satellite&zoom=17&center="
    MyUrl = base + urllib.parse.quote_plus(Loc) + key #added url encoding
    fi = str(index) + "_sat.jpg"
    urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc, fi))

#List of coordinates for testing
tests = {1: "37.79101665,-122.3991486", #01ST ST \ BUSH ST \ MARKET ST
         2: "37.78771761,-122.3950078", #01ST ST \ CLEMENTINA ST
         3: "37.81437339484346,-122.3583278940174", #No image within 50m (radius of 60m finds an image)
         4: "37.79005267,-122.3979386"} #01ST ST \ END
         #Note: Use something like

#db = tests

#Actual list of coordinates
db = LatLongData.getCoords()

for loc in db:
    getStreetViewImage(Loc=db[loc], SaveLoc=myloc, index = loc)
