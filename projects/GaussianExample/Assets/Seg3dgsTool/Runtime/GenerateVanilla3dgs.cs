using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

namespace Seg3dgsTool.Runtime
{
    public class GenerateVanilla3dgs : MonoBehaviour
    {
        public bool IsRunning { get; private set; }
        public string CurrentStatus { get; private set; }
        public string m_ColmapPath;

        public GenerateVanilla3dgs()
        {
            IsRunning = false;
            CurrentStatus = "Idle";
        }

        public async void RunWslScriptViaServer()
        {
            if (IsRunning) return;

            string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
            string dataDirWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string dataDirWsl = ConvertWindowsToWslPath(dataDirWindows);

            if (string.IsNullOrEmpty(dataDirWsl)) return;

            IsRunning = true;
            CurrentStatus = "Sending request to WSL server...";
            
            string jsonPayload = $"{{ \"path\": \"{dataDirWsl}\" }}";

            using (var request = new UnityWebRequest("http://localhost:5001/segment", "POST"))
            {
                byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield(); 
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "Server responded successfully.";
                    Debug.Log("Server Response: " + request.downloadHandler.text);
                }
                else
                {
                    CurrentStatus = $"Error: {request.error}";
                    Debug.LogError($"Error sending request: {request.error}\nResponse: {request.downloadHandler.text}");
                }
            }

            IsRunning = false;
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
    }
}