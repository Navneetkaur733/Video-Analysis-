import numpy as np
import easyocr
import cv2

to_grey = lambda img : cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

class OCR:
    
    def __init__(self, lang=['en'], gpu=True):
        self.reader = easyocr.Reader(lang, gpu=gpu)
    
    def extract_text(self, img, is_grey=False, extra_info=False):
        img = img if is_grey else to_grey(img)
        res = self.reader.readtext(img)
        return res if extra_info else '\n'.join([t for (_, t, __) in res])

    def read_number_plate(self, img):
        txt = self.extract_text(img)
        return txt if len(txt) <= 12 else None