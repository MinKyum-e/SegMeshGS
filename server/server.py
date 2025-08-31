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
        encoding='utf-8', # 인코딩 명시
        bufsize=1
    )

    previous_line_prefix = None # 이전 줄의 앞 5글자를 저장할 변수

    # 로그 파일을 추가 모드('a')로 엽니다.
    with open(LOG_FILENAME, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\n--- New Process Started: {' '.join(command)} ---\n")

        for line in iter(proc.stdout.readline, ''):
            # 1. 클라이언트로 원본 데이터를 스트리밍 (기존 기능 유지)
            # 이 부분이 있어야 Unity에서 실시간으로 데이터를 받습니다.
            yield f"data: {line.strip()}\n\n"

            # 2. 서버 측 로깅 및 콘솔 출력을 위한 처리
            clean_line = line.strip()
            if not clean_line: # 빈 줄은 건너뜁니다.
                continue

            current_line_prefix = clean_line[:5] # 현재 줄의 앞 5글자

            # --- 요청하신 핵심 로직 ---
            # 이전 줄과 현재 줄의 접두사가 같은지 확인 (단, 5글자 이상일 때)
            if len(current_line_prefix) == 5 and current_line_prefix == previous_line_prefix:
                # 같으면: 로그 파일에 저장하고, 콘솔에는 겹쳐서 출력
                log_file.write(clean_line + '\n')
                print(clean_line, end='\r', flush=True) # '\r'로 커서를 맨 앞으로 이동
            else:
                # 다르면: 로그 파일에 저장하지 않고(pass), 콘솔에는 줄바꿈하여 출력
                print(clean_line, end='\n', flush=True)

            # 다음 반복을 위해 현재 줄의 접두사를 '이전 접두사'로 업데이트
            previous_line_prefix = current_line_prefix

    # 프로세스 종료 후 깔끔한 출력을 위해 줄바꿈
    print("\nProcess finished.")

    proc.stdout.close()
    return_code = proc.wait()
    if return_code:
        error_msg = f"error: Process failed with exit code {return_code}"
        print(error_msg)
        yield f"data: {error_msg}\n\n"

@app.route('/segment', methods=['POST'])
def process_data():
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({"error": "Missing 'path' in request"}), 400

    wsl_path = data['path']
    
    if not os.path.isdir(wsl_path):
        return jsonify({"error": f"Directory not found: {wsl_path}"}), 404

    command_args = ['python', 'train_scene.py', '-s', wsl_path]
    
    print(f"Executing command: {' '.join(command_args)}")
    
    return Response(stream_process(command_args), mimetype='text/event-stream')

if __name__ == '__main__':
    # 서버 시작 시 기존 로그 파일이 있으면 삭제하여 깔끔하게 시작
    if os.path.exists(LOG_FILENAME):
        os.remove(LOG_FILENAME)
    app.run(host='0.0.0.0', port=5001, debug=True)

