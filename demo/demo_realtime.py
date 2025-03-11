import cv2
import numpy as np
import torch
import multiprocessing as mp
from PhysNetModel import PhysNet
from utils_sig import *
# from facenet_pytorch import MTCNN
from ultralytics import YOLO

import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# 创建队列用于传递人脸图像
face_queue = mp.Queue(maxsize=400)


def video_capture(face_queue, video_path=0, fps=30):
    cap = cv2.VideoCapture(video_path)
    # mtcnn = MTCNN(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # Train Yolov8n model with WiderFace
    model = YOLO("./yolov8n-face.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 转换帧为RGB并进行人脸检测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # boxes, _ = mtcnn.detect(frame_rgb)

        results = model.predict(source=frame_rgb, device=device, save=False, verbose=False)
        if len(results) == 0:
            continue

        boxes = results[0].boxes.xyxy.tolist()
        if len(boxes) == 0:
            continue

        if boxes is not None:
            box = boxes[0]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box_len = max(x2 - x1, y2 - y1)
            box_half_len = np.round(box_len / 2 * 1.1).astype('int')
            box_mid_y = np.round((y2 + y1) / 2).astype('int')
            box_mid_x = np.round((x2 + x1) / 2).astype('int')

            cropped_face = frame_rgb[
                max(0, box_mid_y - box_half_len): box_mid_y + box_half_len,
                max(0, box_mid_x - box_half_len): box_mid_x + box_half_len
            ]

            # 调整图像尺寸为128x128
            cropped_face = cv2.resize(cropped_face, (128, 128))

            # 如果队列未满，将图像放入队列
            if not face_queue.full():
                try:
                    face_queue.put(cropped_face)
                    frame_count += 1
                except queue.Full:
                    print("Queue is full, skipping frame")
                # 每处理5帧输出进度
                if frame_count % 5 == 0:
                    print(f"Processed {frame_count} frames")

        # 如果未检测到人脸，显示提示
        else:
            cv2.putText(frame, 'No face detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示视频帧
        cv2.imshow('Real-time rPPG', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def heart_rate_estimation(face_queue, window_duration=10, fps=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('./model_weights.pt', map_location=device))

    window_size = int(window_duration * fps)
    rppg_buffer = []
    time_buffer = []
    face_buffer = []

    with torch.no_grad():
        while True:
            # 从队列中获取人脸图像
            if not face_queue.empty():
                # try:
                face = face_queue.get()
                face_buffer.append(face)
                # cv2.imshow('Face', type(face))
                cv2.imwrite('face.jpg', face)
                # except :
                #     continue

            # 当收集到足够帧数时进行心率估计
            if len(face_buffer) == window_size:
                print(f"face_buffer shape: {face_buffer[0].shape}")

                # 转换为Tensor并调整形状
                face_buffer = np.array(face_buffer)  # (N, 128, 128, 3)
                face_buffer = np.transpose(face_buffer, (3, 0, 1, 2))  # (3, N, 128, 128)
                face_tensor = torch.tensor(face_buffer[np.newaxis, ...], dtype=torch.float32).to(device)
                print(f"face_tensor shape: {face_tensor.shape}")

                # 使用模型进行rPPG估计
                rppg = model(face_tensor)[:, -1, :].cpu().numpy().flatten()
                rppg_buffer.extend(rppg.tolist())
                time_buffer.extend(np.arange(len(rppg)) / fps + len(time_buffer) / fps)

                # 计算心率
                filtered_rppg = butter_bandpass(np.array(rppg_buffer), lowcut=0.6, highcut=4, fs=fps)
                hr, psd_y, psd_x = hr_fft(filtered_rppg, fs=fps)

                # 打印心率
                print('Heart rate: %.2f bpm' % hr)

                # 清空缓冲区以便收集新数据
                face_buffer = []

# 加载模型


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # 启动两个进程
    thread1 = mp.Process(target=video_capture, args=(face_queue, 0,))
    thread2 = mp.Process(target=heart_rate_estimation, args=(face_queue,))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

# video_capture(video_path='rtsp://admin:Hbtech2017@192.168.31.150:554/cam/realmonitor?channel=1&subtype=0')
