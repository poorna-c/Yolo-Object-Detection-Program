from imageai.Detection import VideoObjectDetection

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolo.h5")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path="traffic.mp4",
                            output_file_path="traffic_detected_2"
                            , frames_per_second=30, log_progress=True)
print(video_path)