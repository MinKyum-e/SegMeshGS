using UnityEngine;
using System.IO;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Text;
using UnityEngine.Networking;

namespace SegNeshGS.Runtime
{
    public class SegMeshGS : MonoBehaviour
    {
        public string m_VideoPath;

        public bool IsRunning { get; private set; } = false;
        public string CurrentStatus { get; private set; } = "Idle";
        public string m_Query;
        
        private readonly Queue<string> m_LogQueue = new Queue<string>();
        private readonly object m_LogQueueLock = new object();

        
         private async Task<int> RunCommandInShellAsync(string command, [CallerMemberName] string caller="")
        {
            CurrentStatus = $"Starting process... {caller}";
            Debug.Log($"Executing async command in shell: {command}");

            await Task.Run(() =>
            {
                using (var proc = new Process())
                {
                    proc.StartInfo.UseShellExecute = false;
                    proc.StartInfo.RedirectStandardInput = true;
                    proc.StartInfo.RedirectStandardOutput = true;
                    proc.StartInfo.RedirectStandardError = true;
                    proc.StartInfo.CreateNoWindow = true;
                    proc.StartInfo.FileName = "cmd.exe";
                    
                    proc.OutputDataReceived += (sender, args) =>
                    {
                        if (!string.IsNullOrEmpty(args.Data))
                        {
                            lock (m_LogQueueLock)
                            {
                                m_LogQueue.Enqueue(args.Data);
                            }
                        }
                    };
                    proc.ErrorDataReceived += (sender, args) =>
                    {
                        if (!string.IsNullOrEmpty(args.Data))
                        {
                            lock (m_LogQueueLock)
                            {
                                m_LogQueue.Enqueue($"ERROR: {args.Data}");
                            }
                        }
                    };

                    proc.Start();
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();

                    proc.StandardInput.WriteLine(command);
                    proc.StandardInput.WriteLine("exit");
                    proc.StandardInput.Flush();
                    proc.StandardInput.Close();

                    proc.WaitForExit();

                    lock (m_LogQueueLock)
                    {
                        if (proc.ExitCode == 0)
                            m_LogQueue.Enqueue("Process finished successfully.");
                        else
                            m_LogQueue.Enqueue($"Process finished with exit code: {proc.ExitCode}.");
                    }
                }
                
            });
#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
            return 0;
        }
         
        
        public async Task<int> ExtractFramesFromVideo()
        {
            if (string.IsNullOrEmpty(m_VideoPath) || !File.Exists(m_VideoPath))
            {
                Debug.LogError("Video path is not valid. Please select a video file first.");
                return -1;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_VideoPath);
            string outputDirectory = Path.Combine(OutputPath.SourceDataPath, videoName, "input");
            Directory.CreateDirectory(outputDirectory);

            string outputPattern = Path.Combine(outputDirectory, "%04d.jpg");
            string command = $"ffmpeg -y -i \"{m_VideoPath}\" -qscale:v 1 -qmin 1 -vf fps=3 \"{outputPattern}\"";
            
            return await RunCommandInShellAsync(command);
        }
        
        public async Task<int> RunColmapConversion()
        {
            if (string.IsNullOrEmpty(m_VideoPath) || !File.Exists(m_VideoPath))
            {
                Debug.LogError("Video path is not valid. Please select a video file first.");
                return -1;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_VideoPath);
            string colmapProjectDir = Path.Combine(OutputPath.SourceDataPath, videoName);

            if (!Directory.Exists(Path.Combine(colmapProjectDir, "input")))
            {
                Debug.LogError(
                    $"Image directory not found in '{colmapProjectDir}'. Please run 'Extract Frames with FFmpeg' first.");
                return -1;
            }

            string pythonScriptsDir =
                Path.GetFullPath(Path.Combine(Application.dataPath, "PythonScripts"));

            if (!File.Exists(Path.Combine(pythonScriptsDir, "convert.py")))
            {
                Debug.LogError($"Python script 'convert.py' not found in '{pythonScriptsDir}'.");
                return -1;
            }

            string command = $"cd /d \"{pythonScriptsDir}\" && python convert.py -s \"{colmapProjectDir}\"";

            return await RunCommandInShellAsync(command);
        }

        public async Task<int> RunColmapFullPipeline()
        {
            Debug.Log("Starting Full COLMAP Pipeline...");

            int ffmpegResult = await ExtractFramesFromVideo();
            if (ffmpegResult != 0)
            {
                Debug.LogError("FFmpeg frame extraction failed, stopping pipeline.");
                return -1;
            }

            int colmapResult = await RunColmapConversion();
            if (colmapResult != 0)
            {
                Debug.LogError("COLMAP conversion failed.");
                return -1;
            }
            Debug.Log("Full COLMAP Pipeline finished successfully.");
            return 0;
        }
        public async Task<int> RunSegMeshGSPipeline()
        {

            string videoName = Path.GetFileNameWithoutExtension(m_VideoPath);
            string inputFolderWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string inputFolderWsl = ConvertWindowsToWslPath(inputFolderWindows);

            if (string.IsNullOrEmpty(inputFolderWsl))
            {
                Debug.LogError("Input Folder Path is not set. Please check the inspector.");
                return -1;
            }

            CurrentStatus = "Sending Extract Mesh request to server...";

            string jsonPayload = $"{{\"input_folder\": \"{inputFolderWsl}\"}}";
            
            using (var request = new UnityWebRequest("http://localhost:5001/segmeshgs", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for SegMeshGSPipeline server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "Extract SegMeshGSPipeline server responded successfully.";
                    Debug.Log("Server Response: " + request.downloadHandler.text);
                }
                else
                {
                    CurrentStatus = $"Error: {request.error}";
                    Debug.LogError($"Error sending request: {request.error}\nResponse: {request.downloadHandler.text}");
                    return -1;
                }
            }

            return 0;
        }
        

        private string ConvertWindowsToWslPath(string windowsPath)
        {
            if (string.IsNullOrEmpty(windowsPath))
                return null;

            // 경로가 절대 경로인지 확인합니다.
            if (windowsPath.Length < 2 || windowsPath[1] != ':')
            {
                Debug.LogError($"Invalid Windows path format: {windowsPath}");
                return null;
            }

            char driveLetter = char.ToLower(windowsPath[0]);
            string restOfPath = windowsPath.Substring(2).Replace('\\', '/');

            return $"/mnt/{driveLetter}{restOfPath}";
        }
        
        public async Task<int> RunSegMeshGSFullPipeline()
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return -1;
            }
            IsRunning = true;
            
            Debug.Log("Starting Full SegMeshGSFull Pipeline...");

            int result = await RunColmapFullPipeline();
            if (result != 0)
            {
                Debug.LogError("colmap failed, stopping pipeline.");
                IsRunning = false;
                return -1;
            }
            result = await RunSegMeshGSPipeline();
            if (result != 0)
            {
                Debug.LogError("SegMeshPipeline failed, stopping pipeline.");
                IsRunning = false;
                return -1;
            }
            
            
            
            Debug.Log("Full SegMeshGSFull Pipeline finished successfully.");
            IsRunning = false;
            return 0;
        }
        
    }
}