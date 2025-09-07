using System.IO;
using System.Threading.Tasks;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Seg3dgsTool.Runtime
{
    public class Segment3dgs : MonoBehaviour
    {
        public enum DownsampleFactor
        {
            x1 = 1,
            x2 = 2,
            x4 = 4,
            x8 = 8
        }

        public bool IsRunning { get; private set; }
        public string CurrentStatus { get; private set; }

        [Header("Paths & Settings")]
        public string m_ColmapPath;
        [Tooltip("Downsample factor for the images.")]
        public DownsampleFactor m_Downsample = DownsampleFactor.x4;
        [Header("Contrastive Training Settings")]
        public int m_Iterations = 10000;
        public int m_NumSampledRays = 1000;


        public Segment3dgs()
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

            using (var request = new UnityWebRequest("http://localhost:5001/vanila", "POST"))
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
        
        public async void RunSamExtractionViaServer()
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
            string imageRootDirWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string imageRootDirWsl = ConvertWindowsToWslPath(imageRootDirWindows);

            IsRunning = true;
            CurrentStatus = "Sending SAM extraction request to server...";

            string jsonPayload = $"{{\"image_root\": \"{imageRootDirWsl}\", \"downsample\": {(int)m_Downsample}}}";

            using (var request = new UnityWebRequest("http://localhost:5001/extract_masks", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for SAM server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "SAM Server responded successfully.";
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

        public async void RunGetScaleViaServer()
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
            string imageRootDirWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string imageRootDirWsl = ConvertWindowsToWslPath(imageRootDirWindows);

            if (string.IsNullOrEmpty(imageRootDirWsl)) return;

            IsRunning = true;
            CurrentStatus = "Sending get_scale request to server...";

            string jsonPayload = $"{{\"image_root\": \"{imageRootDirWsl}\"}}";

            using (var request = new UnityWebRequest("http://localhost:5001/get_scale", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for get_scale server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "Get_scale server responded successfully.";
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

        public async void RunContrastiveTrainingViaServer()
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
            string imageRootDirWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string imageRootDirWsl = ConvertWindowsToWslPath(imageRootDirWindows);

            if (string.IsNullOrEmpty(imageRootDirWsl)) return;

            IsRunning = true;
            CurrentStatus = "Sending contrastive training request to server...";

            string jsonPayload = $"{{\"image_root\": \"{imageRootDirWsl}\", \"iterations\": {m_Iterations}, \"num_sampled_rays\": {m_NumSampledRays}}}";

            using (var request = new UnityWebRequest("http://localhost:5001/train_contrastive", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for contrastive training server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "Contrastive training server responded successfully.";
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

        public async void RunSagaGuiViaServer()
        {
            if (IsRunning)
            {
                Debug.LogWarning("A process is already running.");
                return;
            }

            string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
            string rootFolderWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
            string rootFolderWsl = ConvertWindowsToWslPath(rootFolderWindows);

            if (string.IsNullOrEmpty(rootFolderWsl)) return;
            
            Debug.Log(rootFolderWsl);

            IsRunning = true;
            CurrentStatus = "Sending SAGA GUI request to server...";

            string jsonPayload = $"{{\"root_folder\": \"{rootFolderWsl}\"}}";

            using (var request = new UnityWebRequest("http://localhost:5001/saga_gui", "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");

                var asyncOp = request.SendWebRequest();

                while (!asyncOp.isDone)
                {
                    CurrentStatus = $"Waiting for SAGA GUI server response... (Upload: {request.uploadProgress * 100:F0}%)";
                    await Task.Yield();
                }

                if (request.result == UnityWebRequest.Result.Success)
                {
                    CurrentStatus = "SAGA GUI server responded successfully.";
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