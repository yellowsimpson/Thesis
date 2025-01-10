import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드
detection_model = YOLO('C://Users//shims//Desktop//github//thesis//YOLO_DEC_NCNN//runs//detect//train4//weights//best.pt')
segmentation_model = YOLO('C://Users//shims//Desktop//github//thesis//YOLO_SEG_NCNN//weights//yolov8//both//1004//weights//best.pt')

# 이미지 경로 설정
image_path = 'C://Users//shims//Desktop//github//thesis//test1.jpg'

# Detection 결과
detection_results = detection_model(image_path)
detection_plot = detection_results[0].plot()

# Segmentation 결과
segmentation_results = segmentation_model(image_path)
segmentation_plot = segmentation_results[0].plot()

# 두 이미지를 같은 크기로 맞추기
height, width, _ = detection_plot.shape
segmentation_plot_resized = cv2.resize(segmentation_plot, (width, height))

# 알파 블렌딩을 통해 이미지를 겹치기
# alpha=0.5, beta=0.5, gamma=0으로 지정 (원하는 값을 조절해보세요)
overlapped_image = cv2.addWeighted(detection_plot, 0.5, segmentation_plot_resized, 0.5, 0)

# 결과를 하나의 창에서 겹친 형태로 확인
cv2.imshow("Overlapped Detection & Segmentation", overlapped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
