import cv2
import numpy as np
import matplotlib.pyplot as plt
from local_utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json
import glob
import pytesseract
from PIL import Image


def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)


wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def get_plate_number(img):
    return pytesseract.image_to_string(img)


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


img = glob.glob("test_images/2.JPG")


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _, LpImg, _, cor = detect_lp(
        wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor


test_image = img[0]
LpImg, cor = get_plate(test_image)


plt.figure(figsize=(12, 5))
plt.axis(False)
plt.imshow(LpImg[0])
plt.savefig("plate.jpg", dpi=150)
img = Image.open('plate.jpg')
img = img.convert('1', dither=Image.NONE)
print(pytesseract.image_to_string(img))

img.show()


# import numpy as np
# import cv2
# from PIL import Image
# import pytesseract as tess


# def ratioCheck(area, width, height):
#     ratio = float(width) / float(height)
#     if ratio < 1:
#         ratio = 1 / ratio
#     if (area < 1063.62 or area > 73862.5) or (ratio < 3 or ratio > 6):
#         return False
#     return True


# def isMaxWhite(plate):
#     avg = np.mean(plate)
#     if(avg >= 115):
#         return True
#     else:
#         return False


# def ratio_and_rotation(rect):
#     (x, y), (width, height), rect_angle = rect
#     if(width > height):
#         angle = -rect_angle
#     else:
#         angle = 90 + rect_angle
#     if angle > 15:
#         return False
#     if height == 0 or width == 0:
#         return False
#     area = height*width
#     if not ratioCheck(area, width, height):
#         return False
#     else:
#         return True


# def clean2_plate(plate):
#     gray_img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray_img, 110, 255, cv2.THRESH_BINARY)
#     if cv2.waitKey(0) & 0xff == ord('q'):
#         pass
#     num_contours, hierarchy = cv2.findContours(
#         thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     if num_contours:
#         contour_area = [cv2.contourArea(c) for c in num_contours]
#         max_cntr_index = np.argmax(contour_area)
#         max_cnt = num_contours[max_cntr_index]
#         max_cntArea = contour_area[max_cntr_index]
#         x, y, w, h = cv2.boundingRect(max_cnt)
#         if not ratioCheck(max_cntArea, w, h):
#             return plate, None
#         final_img = thresh[y:y+h, x:x+w]
#         return final_img, [x, y, w, h]
#     else:
#         return plate, None


# img = cv2.imread("Plate_examples/1.JPG")
# print("Number  input image...",)
# cv2.imshow("input", img)
# if cv2.waitKey(0) & 0xff == ord('q'):
#     pass
# img2 = cv2.GaussianBlur(img, (3, 3), 0)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# img2 = cv2.Sobel(img2, cv2.CV_8U, 1, 0, ksize=3)
# _, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
# morph_img_threshold = img2.copy()
# cv2.morphologyEx(src=img2, op=cv2.MORPH_CLOSE,
#                  kernel=element, dst=morph_img_threshold)
# num_contours, hierarchy = cv2.findContours(
#     morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(img2, num_contours, -1, (0, 255, 0), 1)
# for i, cnt in enumerate(num_contours):
#     min_rect = cv2.minAreaRect(cnt)
#     if ratio_and_rotation(min_rect):
#         x, y, w, h = cv2.boundingRect(cnt)
#         plate_img = img[y:y+h, x:x+w]
#         print("Number  identified number plate...")
#         cv2.imshow("num plate image", plate_img)
#         if cv2.waitKey(0) & 0xff == ord('q'):
#             pass
#         if(isMaxWhite(plate_img)):
#             clean_plate, rect = clean2_plate(plate_img)
#             if rect:
#                 fg = 0
#                 x1, y1, w1, h1 = rect
#                 x, y, w, h = x+x1, y+y1, w1, h1
#                 # cv2.imwrite("clena.png",clean_plate)
#                 plate_im = Image.fromarray(clean_plate)
#                 text = tess.image_to_string(plate_im, lang='eng')
#                 print("Number  Detected Plate Text : ", text)
