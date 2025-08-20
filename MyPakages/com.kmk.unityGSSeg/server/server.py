# server.py
# 필요한 라이브러리: pip install flask ultralytics opencv-python torch
from flask import Flask, request, jsonify
import numpy as np
import cv2
from ultralytics import SAM
import base64
import torch

# --- SAM 모델 로드 (서버 시작 시 한 번만 실행) ---
try:
    model = SAM('sam2.1_b.pt')
    print("Ultralytics SAM2 모델 로딩 성공!")
except Exception as e:
    print(f"모델 로딩 실패: {e}")
    model = None

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def segment_and_colorize():
    if model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500

    # 1. Unity에서 보낸 이미지 데이터 수신
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 없습니다.'}), 400

    file = request.files['image']
    image_bytes = file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 2. SAM 모델로 이미지의 모든 객체 분할
    results = model(img_np, verbose=False)

    if not results or results[0].masks is None:
        # 분할된 객체가 없으면 에러 메시지와 함께 원본 이미지를 반환
        _, buffer = cv2.imencode('.png', img_np)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': img_base64, 'error': 'No objects segmented.'})

    # 3. 각 마스크를 다른 색상으로 칠하기
    # 원본 이미지와 같은 크기의 검은색 오버레이 이미지를 생성
    overlay = np.zeros_like(img_np, dtype=np.uint8)

    all_masks_tensor = results[0].masks.data

    for mask_tensor in all_masks_tensor:
        # 각 마스크에 대한 무작위 BGR 색상 생성
        color = np.random.randint(0, 255, (1, 3)).tolist()[0]

        # 마스크 텐서를 Numpy boolean 배열로 변환
        mask_np = mask_tensor.cpu().numpy().astype(bool)

        # 오버레이 이미지의 해당 영역에 색상을 적용
        overlay[mask_np] = color

    # 4. 원본 이미지와 색상 오버레이를 자연스럽게 합성
    alpha = 0.6  # 투명도
    color_segmented_image = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)

    # 5. 최종 이미지를 Base64로 인코딩하여 반환
    _, buffer = cv2.imencode('.png', color_segmented_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_base64})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)