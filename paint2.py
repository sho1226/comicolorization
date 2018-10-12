import argparse
import json
import os
from PIL import Image
import sys

import threading
import numpy
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
parser.add_argument('--input_image', default='./sample/test.jpg', help='path of input page image')
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
#====================================================================================
#着色時参照画像読み込み
reference_images = [Image.open(args.reference_image).convert('RGB')]
#枠の設定(元が漫画なので必要だった)
rect = [[0,0,WIDTH,HEIGHT],]
#====================================================================================
#着色処理
"""
class DrawThread(threading.Thread):
	def __init__(self, frame):
		super(DrawThread, self).__init__()
		self._frame = frame

	def run(self):
		#グレースケール化
		self._frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
		#2値化
		ret,self._frame = cv2.threshold(self._frame,110,255,cv2.THRESH_BINARY)
		#グレースケール化で減った次元を直す
		self._frame=self._frame.reshape((WIDTH,HEIGHT,1))
		self._frame = numpy.concatenate([self._frame,self._frame,self._frame],axis=2)
		#OpenVCからPILに変換
		self._frame=Image.fromarray(self._frame)
		#各変数設定
		pipelines = pipeline.PagePipeline( \
			drawer=drawer, \
			drawer_sr=drawer_sr, \
			image=self._frame, \
			reference_images=reference_images, \
			threshold_binary=190, \
			threshold_line=130, \
			panel_rects=rect, \
		)
		#着色
		print("draw_start :")
		start = time.time()
		#drawn_image = pipelines.process()
		print("time >> ",time.time()-start)

"""
def Draw_img(_frame):
	_width = _frame.shape[0]
	_height = _frame.shape[1]
	#グレースケール化
	_frame = cv2.cvtColor(_frame,cv2.COLOR_RGB2GRAY)
	#2値化
	ret,_frame = cv2.threshold(_frame,110,255,cv2.THRESH_BINARY)
	#グレースケール化で減った次元を直す
	_frame=_frame.reshape((_width,_height,1))
	_frame = numpy.concatenate([_frame,_frame,_frame],axis=2)
	#OpenVCからPILに変換
	_frame=Image.fromarray(_frame)
	#各変数設定
	pipelines = pipeline.PagePipeline( \
		drawer=drawer, \
		drawer_sr=drawer_sr, \
		image=_frame, \
		reference_images=reference_images, \
		threshold_binary=190, \
		threshold_line=130, \
		panel_rects=rect, \
	)
	#着色
	start = time.time()
	print("draw start")
	drawn_image = pipelines.process()
	print("draw end. time >> ",time.time()-start)
	return drawn_image


if __name__=="__main__":
	cap = cv2.VideoCapture(1)
	cap.set(3,WIDTH) #width
	cap.set(4,HEIGHT) #height

	ret, frame = cap.read()
	while cap.isOpened():
		#キャプチャ
		ret, frame = cap.read()
		Draw_img(frame)
		cv2.imshow('camera capture', frame)
		k = cv2.waitKey(10) # 1msec待つ
		if k == 27: # ESCキーで終了
			break

	cap.release()
	cv2.destroyAllWindows()
