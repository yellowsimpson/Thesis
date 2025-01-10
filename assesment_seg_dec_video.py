import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드
detection_model = YOLO('C://Users//shims//Desktop//github//thesis//YOLO_DEC_NCNN//runs//detect//train4//weights//best.pt')
segmentation_model = YOLO('C://Users//shims//Desktop//github//thesis//YOLO_SEG_NCNN//weights//yolov8//both//1004//weights//best.pt')

# 동영상 파일 경로 혹은 웹캠(0) 설정
video_path = 'C://Users//shims//Desktop//github//thesis//test1.AVI'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 동영상이 끝났을 경우 혹은 에러 발생 시 종료

    # ------------------------------------------
    # 1) Detection
    # ------------------------------------------
    detection_results = detection_model(frame)
    detection_plot = detection_results[0].plot()

    # ------------------------------------------
    # 2) Segmentation
    # ------------------------------------------
    segmentation_results = segmentation_model(frame)
    segmentation_plot = segmentation_results[0].plot()

    # ------------------------------------------
    # 3) 두 이미지를 같은 크기로 맞춘 뒤 겹치기
    # ------------------------------------------
    # detection_plot의 크기를 기준으로 resize
    height, width, _ = detection_plot.shape
    seg_plot_resized = cv2.resize(segmentation_plot, (width, height))

    # 알파 블렌딩(0.5, 0.5)으로 두 이미지를 겹침
    overlapped = cv2.addWeighted(detection_plot, 0.5, seg_plot_resized, 0.5, 0)

    # ------------------------------------------
    # 4) 하나의 창에 표시
    # ------------------------------------------
    cv2.imshow("Overlapped Detection & Segmentation", overlapped)

    # ESC(27) 키를 누르면 종료 (원하는 키로 변경 가능)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ------------------------------------------
# 자원 해제
# ------------------------------------------
cap.release()
cv2.destroyAllWindows()
