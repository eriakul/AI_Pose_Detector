import cv2
import numpy as np
import requests
import pandas as pd
from urllib.request import urlopen 
from pickle import dump, load
from os.path import exists


# # Define a list of file dependencies and where to download them
# dependencies = {
#   "image_urls.csv": "https://drive.google.com/uc?id=1sWSoaobb5IYemJqHi5hpo3AD_esfZR-0",
# }
# for filename, url in dependencies.items():
#   r = requests.get(url)
#   with open(filename, 'wb') as f:
#     f.write(r.content)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Mother:
    def __init__(self, isFirstTime = False):
        self.current_index = 0 if isFirstTime else self.get_archived_index()
        self.image_urls = self.get_image_urls()

        self.current_image = self.load_image(self.current_index)
        self.crops = []

        self.cropping = False

        self.crop_dimensions = dict(x_start = None, y_start = None, x_end = None, y_end = None)
        self.rectangles = []

        self.last_key = None

        print("Starting Mother on image {}.".format(self.current_index))


    def handle_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.crop_dimensions["x_start"], self.crop_dimensions["y_start"], self.crop_dimensions["x_end"], self.crop_dimensions["y_end"] = x, y, x, y
            self.cropping = True
    
        # Mouse is Moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping == True:
                self.crop_dimensions["x_end"], self.crop_dimensions["y_end"] = x, y
    
        # if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.crop_dimensions["x_end"], self.crop_dimensions["y_end"] = x, y
            self.cropping = False # cropping is finished
            refPoint = [(self.crop_dimensions["x_start"], self.crop_dimensions["y_start"]), (self.crop_dimensions["x_end"], self.crop_dimensions["y_end"])]
            if len(refPoint) == 2 and None not in refPoint[0] and None not in refPoint[1]:
                cropped_image = self.current_image[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
                self.add_crop(cropped_image)
                self.add_rect(refPoint)
                print("{} images captured.".format(len(self.rectangles)))
                

    def add_crop(self, im):
        if im.shape[0] == 0 or im.shape[1] == 0:
            print("Invalid shape: {}".format(im.shape[:2]))
            return
        self.crops.append(im)
        self.crop_dimensions = dict(x_start = None, y_start = None, x_end = None, y_end = None)

    def add_rect(self, ref):
        self.rectangles.append(ref)

    def save_images_from_crops(self):
        target = 200
        images = []
        for crop in self.crops:
            resize = self.resize_image(crop, target_size = target)
            final = self.paste_in_square(resize, target_size = target)
            if final.shape[:2] != [target, target]:
                print("Image shape: {}".format(final.shape[:2]))
            images.append(final)
        self.save_images(images)

    
    def load_image(self,index):
        url = self.image_urls[index]
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    
    def save_images(self, images):
        for i, im in enumerate(images):
            file_path = 'C:\\Users\\elu\\Documents\\_Code\Machine Learning Final\\AI_Pose_Detector\\data\\{}-{}.jpg'.format(self.current_index, i)
            cv2.imwrite(file_path, im)
            if exists(file_path):
                print("Successfully saved crop {} of image {}.".format(i, self.current_index))
            else:
                print("SAVE FAILED: crop {} of image {}.".format(i, self.current_index))

    def next(self):
        self.save_images_from_crops()
        self.crop_dimensions = dict(x_start = None, y_start = None, x_end = None, y_end = None)
        self.rectangles = []
        self.crops = []
        self.current_index += 1
        dump(self.current_index,open( "archived_index.p", "wb" ))
        self.current_image = self.load_image(self.current_index)

    def get_archived_index(self):
        return load(open( "archived_index.p", "rb" ))

    def get_image_urls(self):
        urls = pd.read_csv("image_urls.csv") 
        images = urls["image_url"]
        return images

    def resize_image(self, im, target_size = 200):
        dimensions = im.shape[:2]
        larger_side = dimensions.index(max(dimensions))
        scale_factor = target_size/dimensions[larger_side]
        new_dimensions = [0,0]
        new_dimensions[larger_side-1] = target_size
        new_dimensions[larger_side] = int(scale_factor*dimensions[larger_side-1])
        resized = cv2.resize(im, tuple(new_dimensions), interpolation = cv2.INTER_AREA)
        return resized

    def paste_in_square(self, im, target_size = 200):
        white = np.full((target_size, target_size, 3), 255, np.uint8) 
        height, width = im.shape[:2]
        offsetH = int((target_size-height)/2)
        offsetW = int((target_size-width)/2)
        white[offsetH:offsetH+height, offsetW:offsetW+width] = im
        return white



 
cv2.namedWindow("Cropping Tool")


mother = Mother()
cv2.setMouseCallback("Cropping Tool", mother.handle_event)

while True:
 
    image = mother.current_image.copy()

    # draw rectangles
    for r in mother.rectangles:
        ref1, ref2 = r
        cv2.rectangle(image, ref1, ref2, (255, 200, 0), 2)
    if mother.cropping:
        current_rect = [(mother.crop_dimensions["x_start"], mother.crop_dimensions["y_start"]), (mother.crop_dimensions["x_end"], mother.crop_dimensions["y_end"])]
        cv2.rectangle(image, current_rect[0], current_rect[1], (255, 200, 0), 2)


    cv2.imshow("Cropping Tool", image)
    key = cv2.waitKey(1)

    if key == 32 or key == 110 or key==113 or key == 122:
        if key == 32:
            if mother.last_key == 110:
                mother.next()
            if mother.last_key == 113:
                cv2.destroyAllWindows()
                break
            if mother.last_key == 122:
                mother.crops.pop(-1)
                mother.rectangles.pop(-1)
        mother.last_key = key
 
# close all open windows
cv2.destroyAllWindows()