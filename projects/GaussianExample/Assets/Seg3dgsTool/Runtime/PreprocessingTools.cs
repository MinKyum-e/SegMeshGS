using UnityEngine;
using System.IO;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Seg3dgsTool.Runtime
{
    public class PreprocessingTools : MonoBehaviour
    {
        public string m_VideoPath;

        public bool IsRunning { get; private set; } = false;
        public string CurrentStatus { get; private set; } = "Idle";

        private readonly Queue<string> m_LogQueue = new Queue<string>();
        private readonly object m_LogQueueLock = new object();

        public void ExtractFramesFromVideo()
        {
            if (string.IsNullOrEmpty(m_VideoPath) || !File.Exists(m_VideoPath))
            {
                Debug.LogError("Video path is not valid. Please select a video file first.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_VideoPath);
            string outputDirectory = Path.Combine(OutputPath.SourceDataPath, videoName, "input");
            Directory.CreateDirectory(outputDirectory);

            string outputPattern = Path.Combine(outputDirectory, "%04d.jpg");
            string command = $"ffmpeg -y -i \"{m_VideoPath}\" -qscale:v 1 -qmin 1 -vf fps=3 \"{outputPattern}\"";
            
            RunCommandInShell(command);
        }

        private void RunCommandInShell(string command)
        {
            Debug.Log($"Executing command in shell: {command}");

            using (var proc = new Process())
            {
                proc.StartInfo.UseShellExecute = false;
                proc.StartInfo.RedirectStandardInput = true;
                proc.StartInfo.RedirectStandardOutput = true;
                proc.StartInfo.RedirectStandardError = true;
                proc.StartInfo.CreateNoWindow = true;
                proc.StartInfo.FileName = "cmd.exe";

                proc.Start();

                proc.StandardInput.WriteLine(command);
                proc.StandardInput.WriteLine("exit");
                proc.StandardInput.Flush();
                proc.StandardInput.Close();

                string output = proc.StandardOutput.ReadToEnd();
                string error = proc.StandardError.ReadToEnd();

                proc.WaitForExit();

                if (!string.IsNullOrEmpty(output))
                    Debug.Log("Shell Output:\n" + output);

                if (proc.ExitCode == 0)
                {
                    Debug.Log($"Process executed successfully.");
                    if (!string.IsNullOrEmpty(error))
                        Debug.Log("Process Info (from stderr):\n" + error);

#if UNITY_EDITOR
                    UnityEditor.AssetDatabase.Refresh();
#endif
                }
                else
                {
                    if (!string.IsNullOrEmpty(error))
                        Debug.LogError($"Process Error (Exit Code: {proc.ExitCode}):\n{error}");
                }
            }
        }

        private async void RunCommandInShellAsync(string command)
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return;
            }

            IsRunning = true;
            CurrentStatus = "Starting process...";
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

            IsRunning = false;
#if UNITY_EDITOR
            UnityEditor.AssetDatabase.Refresh();
#endif
        }
        
        private void Update()
        {
            lock (m_LogQueueLock)
            {
                while (m_LogQueue.Count > 0)
                {
                    string log = m_LogQueue.Dequeue();
                    if (log.StartsWith("ERROR:"))
                        Debug.LogError(log);
                    else
                        Debug.Log(log);
                }
            }
        }



        public void RunColmapConversion()
        {
            if (string.IsNullOrEmpty(m_VideoPath) || !File.Exists(m_VideoPath))
            {
                Debug.LogError("Video path is not valid. Please select a video file first.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_VideoPath);

            // COLMAP 프로젝트 폴더 (내부에 'input' 폴더가 있어야 함)
            string colmapProjectDir = Path.Combine(OutputPath.SourceDataPath, videoName);

            if (!Directory.Exists(Path.Combine(colmapProjectDir, "input")))
            {
                Debug.LogError(
                    $"Image directory not found in '{colmapProjectDir}'. Please run 'Extract Frames with FFmpeg' first.");
                return;
            }

            // Python 스크립트가 있는 폴더 경로 (Assets/Seg3dgsTool/PythonScripts)
            string pythonScriptsDir =
                Path.GetFullPath(Path.Combine(Application.dataPath, "Seg3dgsTool", "PythonScripts"));

            if (!File.Exists(Path.Combine(pythonScriptsDir, "convert.py")))
            {
                Debug.LogError($"Python script 'convert.py' not found in '{pythonScriptsDir}'.");
                return;
            }

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            string command = $"cd /d \"{pythonScriptsDir}\" && python convert.py -s \"{colmapProjectDir}\"";
#else
             string command = $"cd \"{pythonScriptsDir}\"; python convert.py -s \"{colmapProjectDir}\"";
#endif
            RunCommandInShellAsync(command);
        }
    }
}
