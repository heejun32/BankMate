from matplotlib import pyplot as plt
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import ImageFont, ImageDraw
from PIL import Image
import cv2
import time
import dlib
import numpy as np

# 로그인 및 서류 등록시 바로 얼굴 인증을 실시 하는 함수
class FaceCon:  # Face Confirm
    def __init__(self):
        self.face_finder()

    # 얼굴 사진에서 각 끝점을 표시하는 함수
    def face_dots(self, dets):
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

    # 이미지에서 얼굴을 찾는 함수
    def face_finder(self):
        # 타이머 설정
        TIMER = int(0)

        # 한글 폰트 사용 함수
        font_kor = ImageFont.truetype("gulim.ttc", 20)
        
        # dlib 얼굴 탐지 함수
        detector = dlib.get_frontal_face_detector()
        cap = cv2.VideoCapture(0)
        start = 0

        # 웹캠 연결 및 작동
        while(True):
            ret, frame = cap.read()
            if start==0:
                frame = cv2.rectangle(frame, (200, 400), (500, 100), (255, 0, 0),3)
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                draw.text((270, 70), "촬영이 시작됩니다", font=font_kor, fill=(255, 255, 255))
                draw.text((200, 410), "사각형 안에 얼굴을 맞춰주세요\n2초 후 촬영됩니다", font=font_kor, fill=(255, 255, 255))
                img = np.array(img)
                cv2.imshow('Face_Recognition', img)

            else:
                frame = cv2.rectangle(frame, (200, 400), (500, 100), (0, 0, 255),3)
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
                while TIMER >=0:
                    ret, frame = cap.read()
                    img = cv2.rectangle(frame, (200, 400), (500, 100), (11, 201, 4),3)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img, str(TIMER),(320, 90), font, 3, (0, 255, 255),4, cv2.LINE_AA)
                    cv2.imshow('Face_Recognition', img)
                    cv2.waitKey(1)

                    ret, frame = cap.read()    # Read 결과(제대로 읽으면 True, 아니면 False)와 frame
                    img = Image.fromarray(frame)
                    img = np.array(img)

                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    dets = detector(img_gray, 1)
                    left, right, top, bottom = self.face_dots(dets)

                    if 200 < left < 500 and 100 < top < 400 and right < 500 and bottom > 100:
                        cur = time.time()

                        if cur-prev>=1:
                            prev=cur
                            TIMER = TIMER-1

                    else:
                        start=1
                        TIMER = 2
                        break

                # 웹캠 촬영 조건 충족시 사진 저장 경로 설정
                if TIMER < 0:
                    ret, frame = cap.read()
                    save_path = 'E:\Github\Bankmate\database_temp\\temp.jpg'
                    cv2.imwrite(save_path,frame)
                    break

            if cv2.waitKey(1)==27 : break

        cap.release()
        cv2.destroyAllWindows()

class VGGFACE:
    def __init__(self, ID_NUM):
        self.banker = 'E:\Github\Bankmate\database\\' + str(ID_NUM) + '.jpg'
        self.test = 'E:\\Github\\Bankmate\\database_temp\\temp.jpg'
        self.result = self.result(self.banker, self.test)

    def extract_face(self, filename, required_size=(224, 224)):
        pixels = plt.imread(filename)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array

    def get_embeddings(self, filename):
        faces = [self.extract_face(filename)]
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        yhat = model.predict(samples)
        return yhat

    def is_match(self, known_embedding, candidate_embedding, thresh=0.5):
        score = cosine(known_embedding, candidate_embedding)
        if score <= thresh:
            return True
        else:
            return False

    def result(self, banker, test):
        embeddings_banker = self.get_embeddings(banker)
        embeddings_filenames = self.get_embeddings(test)
        return self.is_match(embeddings_banker, embeddings_filenames)