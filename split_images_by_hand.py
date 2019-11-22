import cv2
import numpy as np
import requests
import pandas as pd
from urllib.request import urlopen 
from pickle import dump, load

# Define a list of file dependencies and where to download them
dependencies = {
  "image_urls.csv": "https://drive.google.com/uc?id=1sWSoaobb5IYemJqHi5hpo3AD_esfZR-0",
}
for filename, url in dependencies.items():
  r = requests.get(url)
  with open(filename, 'wb') as f:
    f.write(r.content)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Mother:
    def __init__(self, isFirstTime = False):
        self.current_index = 0 if isFirstTime else self.get_archived_index()
        self.image_urls = self.get_image_urls()
        self.current_image = self.load_image(self.current_index)

    def load_image(index):
        url = self.image_urls[index]
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def next(self):
        dump(self.current_index,open( "archived_index.p", "wb" ))
        self.current_index += 1

    def get_archived_index(self):
        return load(open( "archived_index.p", "rb" ))

    def get_image_urls(self):
        urls = pd.read_csv("image_urls.csv") 
        images = urls["image_url"]
        return images

        



test_url = images[3]

print(test_url)

def download_image(url):
  resp = urlopen(url)
  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  return image

def resize_image(im, target_size = 300):
    dimensions = im.shape[:2]
    larger_side = dimensions.index(max(dimensions))
    scale_factor = target_size/dimensions[larger_side]
    new_dimensions = [0,0]
    new_dimensions[larger_side] = target_size
    new_dimensions[larger_side-1] = int(scale_factor*dimensions[larger_side])
    resized = cv2.resize(im, tuple(new_dimensions), interpolation = cv2.INTER_AREA)
    return resized


image = download_image(test_url)

oriImage = image.copy()

cropping = False
 
x_start, y_start, x_end, y_end = 0, 0, 0, 0

 
def mouse_crop(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping
 
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
 
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
 
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]
 
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)
 
while True:
 
    i = image.copy()
 
    if not cropping:
        cv2.imshow("image", image)
 
    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)
 
    cv2.waitKey(1)
 
# close all open windows
cv2.destroyAllWindows()