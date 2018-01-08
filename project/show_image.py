import cv2
import time
from tensorflow.python.platform import gfile

DIM = (round(1280.0 * 0.5), round(720.0 * 0.5))

class Video:
    def __init__(self, category='-1', interval=0.1):
        self.category = category
        self.interval = interval * 1000
        self.cap = cv2.VideoCapture(0)

    def get_images(self, num, is_show=False):
        n = 0
        imgs = []
        init = time.time() * 1000

        while (True):
            curr = time.time() * 1000
            if curr - init < self.interval:
                continue
            n += 1
            init = curr

            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # Our operations on the frame come here
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

            # perform the actual resizing of the image and show it
            img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
            img_byte = cv2.imencode('.jpg', img)[1].tostring()
            imgs.append(img_byte)

            if is_show:
                cv2.imshow('frame', img)

            if n >= num:
                break

        return imgs

if __name__ == '__main__':
    video = Video('banana', 0.3)
    imgs = video.get_images(is_show=True)
    print(len(imgs))