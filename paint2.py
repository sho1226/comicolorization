import argparse
import json
import os
from PIL import Image
import sys

import threading
import numpy as np
import cv2
import time
from concurrent.futures import ProcessPoolExecutor

ROOT_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(ROOT_PATH)
import comicolorization
import comicolorization_sr
import pipeline
#====================================================================================
parser = argparse.ArgumentParser()

#入出力path設定(カメラの場合使用しない)
parser.add_argument('--input_image', default='./sample/test3.jpg', help='path of input page image')
parser.add_argument('--output', default='./sample/colorized_test.png',help='the path of colorized image.')

#着色時参照pathの設定
parser.add_argument('--reference_image', default='./sample/Belmondo-1.png', help='paths of reference images')

#モデルpathの設定
parser.add_argument('--comicolorizatoin_model_directory', default='./model/comicolorization/',
					help='the trained model directory for the comicolorization task.')
parser.add_argument('--comicolorizatoin_model_iteration', type=int, default=550000,
					help='the trained model iteration for the comicolorization task.')
parser.add_argument('--super_resolution_model_directory', default='./model/super_resolution/',
					help='the trained model directory for the super resolution task.')
parser.add_argument('--super_resolution_model_iteration', type=int, default=80000,
					help='the trained model iteration for the super resolution task.')
#gpuの設定
parser.add_argument('--gpu', type=int, default=0,help='gpu number (-1 means the cpu mode).')
args = parser.parse_args()
#====================================================================================
#ニューラルネットワークの設定
drawer = comicolorization.drawer.Drawer(
	path_result_directory=args.comicolorizatoin_model_directory,
	gpu=args.gpu,
)
drawer.load_model(iteration=args.comicolorizatoin_model_iteration)

drawer_sr = comicolorization_sr.drawer.Drawer(
	path_result_directory=args.super_resolution_model_directory,
	gpu=args.gpu,
	colorization_class=comicolorization_sr.colorization_task.ComicolorizationTask,
)
drawer_sr.load_model(iteration=args.super_resolution_model_iteration)
#====================================================================================
#カメラの横と縦の長さ
WIDTH,HEIGHT = 800,600
#WIDTH,HEIGHT = 550,411
#====================================================================================
input_image = Image.open(args.input_image).convert('RGB')
#着色時参照画像読み込み
reference_images = [Image.open(args.reference_image).convert('RGB')]

#枠の設定(元が漫画なので必要だった)
rect = [[0,0,WIDTH,HEIGHT],]
#====================================================================================

def get_line(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 6) # intをいい感じに調整する
	#at = rm_noise(at)
	return at

def get_line2(img):
	# 8近傍の定義
	neiborhood8 = np.array([[1, 1, 1],
							[1, 1, 1],
							[1, 1, 1]],
							np.uint8)
	img_dilate = cv2.dilate(img, neiborhood8, iterations=1)
	img_diff = cv2.absdiff(img, img_dilate)
	img_diff_not = cv2.bitwise_not(img_diff)
	gray = cv2.cvtColor(img_diff_not, cv2.COLOR_RGB2GRAY)
	_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8) # intをいい感じに調整する
	#at = rm_noise(at)
	return th

def BGR2BINARY(image):
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	h_img, s_img, v_img = cv2.split(hsv)
	#cv2.THRESH_OTSUをフラグに足すと閾値を自動決定してくれます。
	_, thresh_img = cv2.threshold(v_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	return thresh_img

#着色処理
class DrawThread(threading.Thread):
	drawn_image = np.zeros((HEIGHT,WIDTH,3))
	def __init__(self, frame):
		super(DrawThread, self).__init__()
		self._frame = frame

	def run(self):
		thresh_img = get_line(self._frame)
		#グレースケール化で減った次元を直す
		thresh_img = cv2.resize(thresh_img,(WIDTH,HEIGHT)).reshape((HEIGHT,WIDTH,1))
		thresh_img = np.concatenate([thresh_img,thresh_img,thresh_img],axis=2)
		#OpenCVからPILに変換
		thresh_img = Image.fromarray(thresh_img)

		#各変数設定
		pipelines = pipeline.PagePipeline( \
			drawer=drawer, \
			drawer_sr=drawer_sr, \
			image=thresh_img, \
			#image = input_image, \
			reference_images=reference_images, \
			#threshold_binary=190, \
			threshold_binary=100, \
			#threshold_line=130, \
			threshold_line=80, \
			panel_rects=rect, \
		)

		#着色
		print("start draw >>>")
		start = time.time()
		DrawThread.drawn_image = pipelines.process()
		DrawThread.drawn_image.save("test.png")
		#PILからOpenCVに変換
		DrawThread.drawn_image = np.asarray(DrawThread.drawn_image)[:,:,::-1]

		print("time >> ",time.time()-start)
		print("finish draw ...")


if __name__=="__main__":
	cap = cv2.VideoCapture(0)
	cap.set(3,800) #width
	cap.set(4,600) #height

	ret, frame = cap.read()
	drawn_image_tmp = np.zeros((HEIGHT,WIDTH,3))
	while cap.isOpened():
		#キャプチャ
		ret, frame = cap.read()

		if(threading.activeCount() == 1):
			drawn_image_tmp = DrawThread.drawn_image
			th = DrawThread(frame)
			th.start()

		cv2.imshow('drawn_image', drawn_image_tmp)
		cv2.imshow("camera",get_line2(frame))
		k = cv2.waitKey(10) # 1msec待つ
		if k == 27: # ESCキーで終了
			break

	cap.release()
	cv2.destroyAllWindows()
