from flask import Flask, request, jsonify, Response
import subprocess
import os

app = Flask(__name__)
LOG_FILENAME = "server_process_log.txt"

def stream_process(command):
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )

    previous_line_prefix = None 

    with open(LOG_FILENAME, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- New Process Started: {' '.join(command)} ---\n")

        for line in iter(proc.stdout.readline, ''):
            yield f"data: {line.strip()}\n\n"

            clean_line = line.strip()
            if not clean_line:
                continue

            current_line_prefix = clean_line[:5]

            if len(current_line_prefix) == 5 and current_line_prefix == previous_line_prefix:
                log_file.write(clean_line + '\n')
                print(clean_line, end='\r', flush=True)
            else:
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




@app.route('/clipsam/run_clipsam', methods=['POST'])
def run_clipsam_old():
    data = request.get_json()
    if not data or 'input_folder' not in data or 'query' not in data:
        return jsonify({"error": "Missing 'input_folder' or 'query' in request"}), 400

    input_folder = data['input_folder']
    query = data['query']

    if not os.path.isdir(input_folder):
        return jsonify({"error": f"Input folder not found: {input_folder}"}), 404

    command_args = [
        'python', 'clipsam/clipsam.py',
        '--input_folder', input_folder,
        '--query', query
    ]

    print(f"Executing command: {' '.join(command_args)}")
    return Response(stream_process(command_args), mimetype='text/event-stream')


@app.route('/clipsam/run_fast_clipsam', methods=['POST'])
def run_clipsam_fast():
    data = request.get_json()
    if not data or 'input_folder' not in data or 'query' not in data:
        return jsonify({"error": "Missing 'input_folder' or 'query' in request"}), 400

    input_folder = data['input_folder']
    query = data['query']

    if not os.path.isdir(input_folder):
        return jsonify({"error": f"Input folder not found: {input_folder}"}), 404

    command_args = [
        'python', 'clipsam/faseClipExtract.py',
        '--input_folder', input_folder,
        '--query', query
    ]

    print(f"Executing command: {' '.join(command_args)}")
    return Response(stream_process(command_args), mimetype='text/event-stream')

@app.route('/sugar', methods=['POST'])
def run_full_pipeline():
    data = request.get_json()
    if not data or 'input_folder' not in data or 'segment_targetname' not in data:
        return jsonify({"error": "Missing 'input_folder' or 'segment_targetname' in request"}), 400

    input_folder = data['input_folder']
    segment_targetname = data['segment_targetname']

    if not os.path.isdir(input_folder):
        return jsonify({"error": f"Input folder not found: {input_folder}"}), 404


    # 체크포인트
    gs_output_dir = os.path.join(input_folder, "saga")

    # 명령어 구성
    command_args = [
        'python', 'train_sugar.py',
        '-s', input_folder,
        '-r', 'dn_consistency',
        '--high_poly', 'True',
        '--export_obj', 'True',
        '--gs_output_dir', gs_output_dir,
        '--white_background', 'TRUE',
        '--segment_targetname', segment_targetname, 
    ]

    print(f"Executing command: {' '.join(command_args)}")
    return Response(stream_process(command_args), mimetype='text/event-stream')


if __name__ == '__main__':
    # 서버 시작 시 기존 로그 파일이 있으면 삭제
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
    app.run(host='0.0.0.0', port=5001, debug=True)