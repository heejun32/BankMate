from PIL import ImageFont, ImageDraw
from PIL import Image
import cv2
import time
import dlib
import numpy as np

class FaceReg:
    def __init__(self, ID_NUM):
        self.face_finder(ID_NUM)

    def face_dots(self, dets):                                  # 얼굴 사진에서 각 끝점을 표시하는 함수
        left = 0
        right = 0
        top = 0
        bottom = 0

        for face in dets:
            left = face.left()
            right = face.right()
            top = face.top()
            bottom = face.bottom()
        return left, right, top, bottom

    def face_finder(self, ID_NUM):                                                              # 이미지에서 얼굴을 찾는 함수
        TIMER = int(0)
        font_kor = ImageFont.truetype("gulim.ttc", 20)                                          # 한글 폰트 사용 함수
        detector = dlib.get_frontal_face_detector()
        cap = cv2.VideoCapture(0)
        start = 0

        while (True):
            ret, frame = cap.read()  # Read 결과(제대로 읽으면 True, 아니면 False)와 frame
            if start == 0:
                frame = cv2.rectangle(frame, (200, 400), (500, 100), (255, 0, 0), 3)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((270, 70), "촬영이 시작됩니다", font=font_kor, fill=(255, 255, 255))
                draw.text((200, 410), "사각형 안에 얼굴을 맞춰주세요\n2초 후 촬영됩니다", font=font_kor, fill=(255, 255, 255))
                img = np.array(img)
                cv2.imshow('Face_Recognition', img)

            else:
                frame = cv2.rectangle(frame, (200, 400), (500, 100), (0, 0, 255), 3)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((250, 70), "위치를 확인해 주세요", font=font_kor, fill=(255, 255, 255))
                img = np.array(img)
                cv2.imshow('Face_Recognition', img)

            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = detector(img_gray, 1)
            global left, right, top, bottom
            left, right, top, bottom = self.face_dots(dets)

            if 200 < left < 500 and 100 < top < 400 and right < 500 and bottom > 100:
                cv2.waitKey(1)
                prev = time.time()
                while TIMER >= 0:
                    ret, frame = cap.read()
                    img = cv2.rectangle(frame, (200, 400), (500, 100), (11, 201, 4), 3)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img, str(TIMER), (320, 90), font, 3, (0, 255, 255), 4, cv2.LINE_AA)
                    cv2.imshow('Face_Recognition', img)
                    cv2.waitKey(1)

                    ret, frame = cap.read()  # Read 결과(제대로 읽으면 True, 아니면 False)와 frame
                    img = Image.fromarray(frame)
                    img = np.array(img)

                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    dets = detector(img_gray, 1)
                    left, right, top, bottom = self.face_dots(dets)

                    if 200 < left < 500 and 100 < top < 400 and right < 500 and bottom > 100:
                        cur = time.time()

                        if cur - prev >= 1:
                            prev = cur
                            TIMER = TIMER - 1

                    else:
                        start = 1
                        TIMER = 2
                        break

                if TIMER < 0:
                    ret, frame = cap.read()
                    save_path = 'E:\Github\Bankmate\database\\' + str(ID_NUM) + '.jpg'
                    cv2.imwrite(save_path, frame)
                    break

            if cv2.waitKey(1) == 27: break

        cap.release()
        cv2.destroyAllWindows()