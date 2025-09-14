using System.IO;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;


namespace Seg3dgsTool.Runtime
{
        public class Mesh3dgs : MonoBehaviour
     {
         public bool IsRunning { get; private set; }
         public string CurrentStatus { get; private set; }
 
         [Header("Paths & Settings")]
         public string m_ColmapPath;
         [Tooltip("The text query for ClipSAM.")]
         public string m_Query;
 
         public Mesh3dgs()
         {
             IsRunning = false;
             CurrentStatus = "Idle";
         }
 
         public async void RunClipSam(bool useFastVersion)
         {
             if (IsRunning)
             {
                 Debug.LogWarning("A process is already running.");
                 return;
             }
 
             string videoName = Path.GetFileNameWithoutExtension(m_ColmapPath);
             string inputFolderWindows = Path.Combine(OutputPath.SourceDataPath, videoName);
             string inputFolderWsl = ConvertWindowsToWslPath(inputFolderWindows);
 
             if (string.IsNullOrEmpty(inputFolderWsl) || string.IsNullOrEmpty(m_Query))
             {
                 Debug.LogError("Input Folder Path or Query is not set. Please check the inspector.");
                 return;
             }
 
             IsRunning = true;
             string endpoint = useFastVersion ? "/clipsam/run_fast_clipsam" : "/clipsam/run_clipsam";
             string processName = useFastVersion ? "Fast-ClipSAM" : "ClipSAM";
             CurrentStatus = $"Sending {processName} request to server...";
 
             string jsonPayload = $"{{\"input_folder\": \"{inputFolderWsl}\", \"query\": \"{m_Query}\"}}";
 
             using (var request = new UnityWebRequest($"http://localhost:5001{endpoint}", "POST"))
             {
                 byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
                 request.uploadHandler = new UploadHandlerRaw(bodyRaw);
                 request.downloadHandler = new DownloadHandlerBuffer();
                 request.SetRequestHeader("Content-Type", "application/json");
 
                 var asyncOp = request.SendWebRequest();
 
                 while (!asyncOp.isDone)
                 {
                     CurrentStatus = $"Waiting for {processName} server response... (Upload: {request.uploadProgress * 100:F0}%)";
                     await Task.Yield();
                 }
 
                 if (request.result == UnityWebRequest.Result.Success)
                 {
                     CurrentStatus = $"{processName} server responded successfully.";
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