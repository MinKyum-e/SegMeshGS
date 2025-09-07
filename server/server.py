from flask import Flask, request, jsonify, Response
import subprocess
import os

app = Flask(__name__)

# 로그 파일 이름 지정
LOG_FILENAME = "server_process_log.txt"

def stream_process(command):
    """
    주어진 명령어를 실행하고, 그 출력을 클라이언트로 스트리밍하면서
    서버 측에 조건부 로깅 및 콘솔 출력을 수행합니다.
    """
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )

    previous_line_prefix = None  # 이전 줄의 앞 5글자를 저장할 변수

    # 로그 파일을 추가 모드('a')로 엽니다.
    with open(LOG_FILENAME, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- New Process Started: {' '.join(command)} ---\n")

        for line in iter(proc.stdout.readline, ''):
            # 1. 클라이언트로 원본 데이터를 스트리밍
            yield f"data: {line.strip()}\n\n"

            # 2. 서버 측 로깅 및 콘솔 출력
            clean_line = line.strip()
            if not clean_line:
                continue

            current_line_prefix = clean_line[:5]

            if len(current_line_prefix) == 5 and current_line_prefix == previous_line_prefix:
                # 이전 줄과 접두사가 같을 때 → 로그 파일에 기록
                log_file.write(clean_line + '\n')
                print(clean_line, end='\r', flush=True)
            else:
                # 다를 때 → 로그 파일에는 기록하지 않고 콘솔 줄바꿈 출력
                print(clean_line, end='\n', flush=True)

            previous_line_prefix = current_line_prefix

    print("\nProcess finished.")

    proc.stdout.close()
    return_code = proc.wait()
    if return_code:
        error_msg = f"error: Process failed with exit code {return_code}"
        print(error_msg)
        yield f"data: {error_msg}\n\n"


@app.route('/vanila', methods=['POST'])
def process_data():
    """
    vanila 3dgs 파이프라인 실행 (train_scene.py)
    """
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "Missing 'path' in request"}), 400

    wsl_path = data['path']
    
    if not os.path.isdir(wsl_path):
        return jsonify({"error": f"Directory not found: {wsl_path}"}), 404

    command_args = ['python', 'train_scene.py', '-s', wsl_path]
    
    print(f"Executing command: {' '.join(command_args)}")
    
    return Response(stream_process(command_args), mimetype='text/event-stream')


@app.route('/extract_masks', methods=['POST'])
def extract_masks():
    """
    SAM 기반 Segmentation 마스크 추출 실행
    """
    data = request.get_json()
    if not data or 'image_root' not in data or 'downsample' not in data:
        return jsonify({"error": "Missing one of required params: image_root,  downsample"}), 400

    image_root = data['image_root']
    downsample = str(data['downsample'])

    if not os.path.isdir(image_root):
        return jsonify({"error": f"Image root not found: {image_root}"}), 404

    command_args = [
        'python', 'extract_segment_everything_masks.py',
        '--image_root', image_root,
        '--downsample', downsample
    ]

    print(f"Executing command: {' '.join(command_args)}")

    return Response(stream_process(command_args), mimetype='text/event-stream')



@app.route('/get_scale', methods=['POST'])
def get_scale():
    """
    3DGS 모델 스케일 계산 실행 (get_scale.py)
    """
    data = request.get_json()
    if not data or 'image_root' not in data:
        return jsonify({"error": "Missing one of required params: image_root"}), 400

    image_root = data['image_root']

    if not os.path.isdir(image_root):
        return jsonify({"error": f"Image root not found: {image_root}"}), 404

    command_args = [
        'python', 'get_scale.py',
        '--image_root', image_root,
        '--model_path', image_root + "/SAGA",
    ]

    print(f"Executing command: {' '.join(command_args)}")

    return Response(stream_process(command_args), mimetype='text/event-stream')


@app.route('/train_contrastive', methods=['POST'])
def train_contrastive():
    """ Contrastive Feature 학습 실행 (train_contrastive_feature.py) """
    data = request.get_json()
    if not data or 'image_root' not in data:
        return jsonify({"error": "Missing 'image_root' in request"}), 400

    image_root = data['image_root']
    iterations = str(data.get('iterations', 10000))         # 기본값 10000
    num_sampled_rays = str(data.get('num_sampled_rays', 1000))  # 기본값 1000

    if not os.path.isdir(image_root):
        return jsonify({"error": f"Image root not found: {image_root}"}), 404

    command_args = [
        'python', 'train_contrastive_feature.py',
        '-m', image_root + "/SAGA",
        '--iterations', iterations,
        '--num_sampled_rays', num_sampled_rays
    ]
    print(f"Executing command: {' '.join(command_args)}")
    return Response(stream_process(command_args), mimetype='text/event-stream')



@app.route('/saga_gui', methods=['POST'])
def saga_gui():
    """ SegAnyGaussian GUI 실행 (saga_gui.py) """
    data = request.get_json()
    if not data or 'root_folder' not in data:
        return jsonify({"error": "Missing 'root_folder' in request"}), 400

    root_folder = data['root_folder']
    command_args = [
        'python', 'saga_gui.py',
        '--model_path', root_folder+"/SAGA"
    ]
    print(f"Executing command: {' '.join(command_args)}")
    return Response(stream_process(command_args), mimetype='text/event-stream')

if __name__ == '__main__':
    # 서버 시작 시 기존 로그 파일이 있으면 삭제
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
    app.run(host='0.0.0.0', port=5001, debug=True)
