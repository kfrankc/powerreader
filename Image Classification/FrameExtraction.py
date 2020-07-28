import cv2
import os

video_path = f'{os.getcwd()}\\Recording.mp4'
output_folder = f'{os.getcwd()}\\Image Classification\\Extracted Slides'

capture = cv2.VideoCapture(video_path) 
length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_num in range(0, length, 8):
    success, image = capture.read()
    if(success):
        cv2.imwrite(f'{output_folder}\\Frame {frame_num}.jpg', image)