import cv2
import time


class Video:
    def __init__(self, category='-1', interval=0.1, dim=(1280, 720)):
        self.category = category
        self.interval = interval * 1000
        self.cap = cv2.VideoCapture(0)
        self.dim = dim

    def get_images(self, num, is_show=False, category_name=None, accuracy=None):
        category = 'NO_CAT' if category_name is None else str(category_name)
        accuracy = '0' if accuracy is None else str(accuracy)

        n = 0
        imgs = []
        init = time.time() * 1000
        key_pressed = -1

        while True:
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
            img = cv2.resize(img, self.dim, interpolation=cv2.INTER_AREA)
            img_byte = cv2.imencode('.jpg', img)[1].tostring()

            if is_show:
                cv2.rectangle(img, (0, 0), (self.dim[0], 50), (0, 0, 0), -1)
                cv2.putText(img, "CATEGORY: %s, ACCURACY: %s" % (category, accuracy), (10, 35),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
                cv2.imshow('frame', img)

                key_pressed = cv2.waitKey(1) & 0xFF
                key_pressed = chr(key_pressed) if key_pressed < 255 else -1
                if key_pressed == 'q':
                    cv2.destroyAllWindows()
                    raise GeneratorExit()

            imgs.append((img_byte, key_pressed))

            if n >= num:
                break

        return imgs


if __name__ == '__main__':
    video = Video('banana', 0.3)

    imgs = video.get_images(100, is_show=True, category_name='NO_CAT', accuracy='0.39')
    print(len(imgs))
