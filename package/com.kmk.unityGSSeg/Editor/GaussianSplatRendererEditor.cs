// SPDX-License-Identifier: MIT

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using GaussianSplatting.Runtime;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEditor;
using UnityEditor.EditorTools;
using UnityEditor.PackageManager;
using UnityEngine;
using UnityEngine.Networking;
using GaussianSplatRenderer = GaussianSplatting.Runtime.GaussianSplatRenderer;
using PackageInfo = UnityEditor.PackageManager.PackageInfo;

namespace GaussianSplatting.Editor
{
    [CustomEditor(typeof(GaussianSplatRenderer))]
    [CanEditMultipleObjects]
    public class GaussianSplatRendererEditor : UnityEditor.Editor
    {
        const string kPrefExportBake = "GaussianSplatting.ExportBakeTransform";
        private static int s_TileGridWidth;
        private static int s_TileGridHeight;

        SerializedProperty m_PropAsset;
        SerializedProperty m_PropRenderOrder;
        SerializedProperty m_PropSplatScale;
        SerializedProperty m_PropOpacityScale;
        SerializedProperty m_PropSHOrder;
        SerializedProperty m_PropSHOnly;
        SerializedProperty m_PropSortNthFrame;
        SerializedProperty m_PropRenderMode;
        SerializedProperty m_PropDisplayInfluenced;
        SerializedProperty m_PropPointDisplaySize;
        SerializedProperty m_PropCutouts;
        SerializedProperty m_PropShaderSplats;
        SerializedProperty m_PropShaderComposite;
        SerializedProperty m_PropShaderDebugPoints;
        SerializedProperty m_PropShaderDebugBoxes;
        SerializedProperty m_PropCSSplatUtilities;
        
        // For Segmentation
        SerializedProperty m_PropSegCamera;
        SerializedProperty m_PropTileSize;
        SerializedProperty m_PropMaxFragmentNodes;
        

        bool m_ResourcesExpanded = false;
        int m_CameraIndex = 0;

        bool m_ExportBakeTransform;

        static int s_EditStatsUpdateCounter = 0;

        static HashSet<GaussianSplatRendererEditor> s_AllEditors = new();

        public static void BumpGUICounter()
        {
            ++s_EditStatsUpdateCounter;
        }

        public static void RepaintAll()
        {
            foreach (var e in s_AllEditors)
                e.Repaint();
        }

        public void OnEnable()
        {
            m_ExportBakeTransform = EditorPrefs.GetBool(kPrefExportBake, false);

            m_PropAsset = serializedObject.FindProperty("m_Asset");
            m_PropRenderOrder = serializedObject.FindProperty("m_RenderOrder");
            m_PropSplatScale = serializedObject.FindProperty("m_SplatScale");
            m_PropOpacityScale = serializedObject.FindProperty("m_OpacityScale");
            m_PropSHOrder = serializedObject.FindProperty("m_SHOrder");
            m_PropSHOnly = serializedObject.FindProperty("m_SHOnly");
            m_PropSortNthFrame = serializedObject.FindProperty("m_SortNthFrame");
            m_PropRenderMode = serializedObject.FindProperty("m_RenderMode");
            m_PropDisplayInfluenced = serializedObject.FindProperty("m_displayInfluenced");
            m_PropPointDisplaySize = serializedObject.FindProperty("m_PointDisplaySize");
            m_PropCutouts = serializedObject.FindProperty("m_Cutouts");
            m_PropShaderSplats = serializedObject.FindProperty("m_ShaderSplats");
            m_PropShaderComposite = serializedObject.FindProperty("m_ShaderComposite");
            m_PropShaderDebugPoints = serializedObject.FindProperty("m_ShaderDebugPoints");
            m_PropShaderDebugBoxes = serializedObject.FindProperty("m_ShaderDebugBoxes");
            m_PropCSSplatUtilities = serializedObject.FindProperty("m_CSSplatUtilities");
            
            // For Segmentation
            m_PropSegCamera = serializedObject.FindProperty("segCamera");
            m_PropTileSize = serializedObject.FindProperty("tileSize");
            m_PropMaxFragmentNodes = serializedObject.FindProperty("maxFragmentNodes");

            s_AllEditors.Add(this);
        }

        public void OnDisable()
        {
            s_AllEditors.Remove(this);
        }

        public override void OnInspectorGUI()
        {
            var gs = target as GaussianSplatRenderer;
            if (!gs)
                return;

            serializedObject.Update();

            GUILayout.Label("Data Asset", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(m_PropAsset);

            if (!gs.HasValidAsset)
            {
                var msg = gs.asset != null && gs.asset.formatVersion != GaussianSplatAsset.kCurrentVersion
                    ? "Gaussian Splat asset version is not compatible, please recreate the asset"
                    : "Gaussian Splat asset is not assigned or is empty";
                EditorGUILayout.HelpBox(msg, MessageType.Error);
            }

            EditorGUILayout.Space();
            GUILayout.Label("Render Options", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(m_PropRenderOrder);
            EditorGUILayout.PropertyField(m_PropSplatScale);
            EditorGUILayout.PropertyField(m_PropOpacityScale);
            EditorGUILayout.PropertyField(m_PropSHOrder);
            EditorGUILayout.PropertyField(m_PropSHOnly);
            EditorGUILayout.PropertyField(m_PropSortNthFrame);

            EditorGUILayout.Space();
            GUILayout.Label("Debugging Tweaks", EditorStyles.boldLabel);
            EditorGUILayout.PropertyField(m_PropRenderMode);
            if (m_PropRenderMode.intValue is (int)GaussianSplatRenderer.RenderMode.DebugPoints or (int)GaussianSplatRenderer.RenderMode.DebugPointIndices)
                EditorGUILayout.PropertyField(m_PropPointDisplaySize);
            EditorGUILayout.PropertyField(m_PropDisplayInfluenced);

            EditorGUILayout.Space();
            m_ResourcesExpanded = EditorGUILayout.Foldout(m_ResourcesExpanded, "Resources", true, EditorStyles.foldoutHeader);
            if (m_ResourcesExpanded)
            {
                EditorGUILayout.PropertyField(m_PropShaderSplats);
                EditorGUILayout.PropertyField(m_PropShaderComposite);
                EditorGUILayout.PropertyField(m_PropShaderDebugPoints);
                EditorGUILayout.PropertyField(m_PropShaderDebugBoxes);
                EditorGUILayout.PropertyField(m_PropCSSplatUtilities);
            }
            bool validAndEnabled = gs && gs.enabled && gs.gameObject.activeInHierarchy && gs.HasValidAsset;
            if (validAndEnabled && !gs.HasValidRenderSetup)
            {
                EditorGUILayout.HelpBox("Shader resources are not set up", MessageType.Error);
                validAndEnabled = false;
            }

            if (validAndEnabled && targets.Length == 1)
            {
                EditCameras(gs);
                EditSegmentationGUI(gs);
                EditGUI(gs);
            }
            if (validAndEnabled && targets.Length > 1)
            {
                MultiEditGUI();
            }

            serializedObject.ApplyModifiedProperties();
        }

        void EditCameras(GaussianSplatRenderer gs)
        {
            var asset = gs.asset;
            var cameras = asset.cameras;
            if (cameras != null && cameras.Length != 0)
            {
                EditorGUILayout.Space();
                GUILayout.Label("Cameras", EditorStyles.boldLabel);
                var camIndex = EditorGUILayout.IntSlider("Camera", m_CameraIndex, 0, cameras.Length - 1);
                camIndex = math.clamp(camIndex, 0, cameras.Length - 1);
                if (camIndex != m_CameraIndex)
                {
                    m_CameraIndex = camIndex;
                    gs.ActivateCamera(camIndex);
                }
                // GaussianSplatRendererEditor.cs 파일에 추가할 코드

// ... 기존의 카메라 목록 및 'Activate' 버튼 아래에 추가 ...

// 카메라 사이클링 시작/정지 버튼
                string buttonText = gs.IsCyclingCameras ? "Stop Camera Cycling" : "Start Camera Cycling (1s)";
                if (GUILayout.Button(buttonText))
                {
                    gs.ToggleAnalysisCycle();
                }
                

                if (GUILayout.Button("Show All Cameras"))
                {
                    gs.CreateAllCameras();
                }
                if (GUILayout.Button("Show Original Images"))
                {
                    ShowOriginalImages(gs);
                }
                if (GUILayout.Button("Show Segment Images"))
                {
                    ShowSegmentedImages(gs);
                }
                if (GUILayout.Button("Segment and Colorize (Server)"))
                {
                    SegmentAndColorizeWithSAM(gs);
                }
            }
        }

        void EditSegmentationGUI(GaussianSplatRenderer gs)
        {
            DrawSeparator();
            GUILayout.Label("Tiled-Based Culling / Segmentation", EditorStyles.boldLabel);
            
            EditorGUILayout.PropertyField(m_PropSegCamera, new GUIContent("Target Camera"));
            EditorGUILayout.PropertyField(m_PropTileSize);
            EditorGUILayout.PropertyField(m_PropMaxFragmentNodes);

            if (GUILayout.Button("Execute FindInfluencedCells"))
            {
                gs.FindInfluencedCells();
            }

            GUILayout.Space(5);
            EditorGUILayout.LabelField("Debugging", EditorStyles.boldLabel);

            using (new EditorGUI.DisabledScope(gs.allocatedFragmentNodes == 0))
            {
                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Print Tile Head Pointers to Console"))
                {
                    gs.RequestTileHeadPointersData((data, width, height) =>
                    {
                        if (data == null || data.Length == 0 || width == 0 || height == 0)
                        {
                            Debug.Log("Tile Head Pointers data is not available or empty. Run 'Execute FindInfluencedCells' first.");
                            return;
                        }
                        
                        // 다른 함수에서 사용할 수 있도록 데이터를 캐싱합니다.
                        GaussianSplatAsset.m_windowCell = data.ToArray();
                        s_TileGridWidth = width;
                        s_TileGridHeight = height;
                        
                        var sb = new StringBuilder();
                        sb.AppendLine($"--- Tile Head Pointers (Grid Size: {width}x{height}) ---");
                        GaussianSplatAsset.m_windowCell = data.ToArray();
                        int activeTileCount = 0;
                        for (int y = 0; y < height; y++)
                        {
                            for (int x = 0; x < width; x++)
                            {
                                int index = y * width + x;
                                uint headNodeIndex = GaussianSplatAsset.m_windowCell[index];
                                if (headNodeIndex != 0xFFFFFFFF) // 0xFFFFFFFF is the clear value
                                {
                                    sb.AppendLine($"Tile({x,3},{y,3}): Head Node Index = {headNodeIndex}");
                                    activeTileCount++;
                                }
                            }
                        }
                        sb.AppendLine($"--- Found {activeTileCount} active tiles. ---");
                        Debug.Log(sb.ToString());
                    });
                }
                 
                 if (GUILayout.Button("Save Pointers as Image..."))
                 {
                     if (GaussianSplatAsset.m_windowCell == null || GaussianSplatAsset.m_windowCell.Length == 0 || s_TileGridWidth == 0 || s_TileGridHeight == 0 || GaussianSplatAsset.nodes == null)
                     {
                         EditorUtility.DisplayDialog("Data Not Available", "Please run 'Print Tile Head Pointers' and 'Print Fragment List' first to cache the data.", "OK");
                     }
                     else
                     {
                         string path = EditorUtility.SaveFilePanel("Save Tile Pointers Image", "", "tile_pointers.png", "png");
                         if (!string.IsNullOrEmpty(path))
                         {
                             try
                             {
                                 int width = s_TileGridWidth;
                                 int height = s_TileGridHeight;
                                 var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
                                 var pixels = new Color[width * height];
 
                                 for (int y = 0; y < height; y++)
                                 {
                                     for (int x = 0; x < width; x++)
                                     {
                                         int dataIndex = y * width + x;
                                         // Texture2D의 (0,0)은 좌측 하단이므로, Y축을 뒤집어 이미지 방향을 맞춥니다.
                                         int textureIndex = (height - 1 - y) * width + x;
                                         uint headNodeIndex = GaussianSplatAsset.m_windowCell[dataIndex];

                                         if (headNodeIndex != 0xFFFFFFFF && headNodeIndex < GaussianSplatAsset.nodes.Length)
                                         {
                                             // 셀의 첫 번째 노드를 가져와서 스플랫 인덱스를 찾습니다.
                                             var node = GaussianSplatAsset.nodes[headNodeIndex];
                                             uint splatIndex = node.splatIndex;
                                             // 스플랫 인덱스를 100개 단위로 그룹화하여 색상을 결정합니다.
                                             uint groupIndex = splatIndex / 100;
                                             float hue = (groupIndex % 20) / 20.0f; // 20개의 색상 그룹으로 순환
                                             pixels[textureIndex] = Color.HSVToRGB(hue, 0.95f, 0.95f);
                                         }
                                         else
                                         {
                                             // 활성화되지 않은 셀은 검은색으로 칠합니다.
                                             pixels[textureIndex] = Color.black;
                                         }
                                     }
                                 }
 
                                 texture.SetPixels(pixels);
                                 texture.Apply();
 
                                 byte[] bytes = texture.EncodeToPNG();
                                 System.IO.File.WriteAllBytes(path, bytes);
                                 DestroyImmediate(texture);
 
                                 Debug.Log($"Successfully saved tile pointers image to {path}");
                             }
                             catch (System.Exception e)
                             {
                                 Debug.LogError($"Failed to save tile pointers image: {e.Message}");
                             }
                         }
                     }
                 }
                 EditorGUILayout.EndHorizontal();


                if (GUILayout.Button("Print Fragment List to Console"))
                {
                    gs.RequestFragmentListBufferData(data =>
                    {
                        if (data == null || data.Length == 0)
                        {
                            Debug.Log("Fragment List data is not available. Run 'Execute FindInfluencedCells' first.");
                            return;
                        }

                        var sb = new StringBuilder();
                        uint nodesToPrint = Math.Min(gs.allocatedFragmentNodes, (uint)data.Length);
                        sb.AppendLine($"--- Fragment List (Printing first {Math.Min(nodesToPrint, 1000)} of {gs.allocatedFragmentNodes} allocated nodes) ---");
                        GaussianSplatAsset.nodes = data.ToArray();
                        uint limit = Math.Min(nodesToPrint, 1000);

                        for (int i = 0; i < limit; i++)
                        {
                            var node = GaussianSplatAsset.nodes[i];
                            string nextNodeStr = node.nextNodeIndex == 0xFFFFFFFF ? "NULL" : node.nextNodeIndex.ToString();
                            sb.AppendLine($"Node[{i,4}]: Splat Index = {node.splatIndex,-7} | Next Node = {nextNodeStr}");
                        }

                        if (nodesToPrint > 1000)
                        {
                            sb.AppendLine($"... and {nodesToPrint - 1000} more nodes.");
                        }

                        sb.AppendLine($"--- End of Fragment List ---");
                        Debug.Log(sb.ToString());
                    });
                }

                if (GUILayout.Button("Process and Store Splat-Cell Links"))
                {
                    if (GaussianSplatAsset.m_windowCell == null || GaussianSplatAsset.m_windowCell.Length == 0 ||
                        GaussianSplatAsset.nodes == null || GaussianSplatAsset.nodes.Length == 0)
                    {
                        Debug.LogError("Data not available. Please run the 'Print' buttons first to populate CPU-side data.");
                        return;
                    }

                    int splatCount = gs.splatCount;
                    if (splatCount == 0)
                    {
                        Debug.LogError("Splat count is zero. Cannot process links.");
                        return;
                    }

                    // 대상 배열을 초기화하고 "유효하지 않음" 마커로 채웁니다.
                    GaussianSplatAsset.m_splatcell = new uint[splatCount];
                    for (int i = 0; i < splatCount; ++i)
                    {
                        GaussianSplatAsset.m_splatcell[i] = 0xFFFFFFFF;
                    }

                    int mappedSplatCount = 0;
                    // 각 타일을 순회합니다. 루프 인덱스가 타일의 ID가 됩니다.
                    for (int tileIndex = 0; tileIndex < GaussianSplatAsset.m_windowCell.Length; tileIndex++)
                    {
                        uint currentNodeIndex = GaussianSplatAsset.m_windowCell[tileIndex];

                        // 이 타일에 대한 연결 리스트를 순회합니다.
                        while (currentNodeIndex != 0xFFFFFFFF && currentNodeIndex < GaussianSplatAsset.nodes.Length)
                        {
                            var node = GaussianSplatAsset.nodes[currentNodeIndex];
                            uint splatIndex = node.splatIndex;

                            if (splatIndex < splatCount && GaussianSplatAsset.m_splatcell[splatIndex] == 0xFFFFFFFF)
                            {
                                GaussianSplatAsset.m_splatcell[splatIndex] = (uint)tileIndex;
                                mappedSplatCount++;
                            }
                            currentNodeIndex = node.nextNodeIndex;
                        }
                    }
                    Debug.Log($"Successfully processed splat-cell links. Mapped {mappedSplatCount} of {splatCount} splats to their first-encountered tile.");

                    // 콘솔에 매핑 결과 출력
                    var sb = new StringBuilder();
                    sb.AppendLine($"--- Splat to Cell Mapping (Showing first 1000 mapped splats) ---");
                    int printCount = 0;
                    for (int i = 0; i < splatCount && printCount < 1000; i++)
                    {
                        uint cellId = GaussianSplatAsset.m_splatcell[i];
                        if (cellId != 0xFFFFFFFF)
                        {
                            sb.AppendLine($"Splat[{i,7}]: Cell ID = {cellId}");
                            printCount++;
                        }
                    }
                    if (mappedSplatCount > 1000)
                    {
                        sb.AppendLine($"... and {mappedSplatCount - 1000} more mapped splats.");
                    }
                    sb.AppendLine($"--- End of Mapping ---");
                    Debug.Log(sb.ToString());
                    // 파일로 저장
                    string path = EditorUtility.SaveFilePanel("Save Splat-Cell Mapping", "", "splat_cell_mapping.txt", "txt");
                    if (!string.IsNullOrEmpty(path))
                    {
                        try
                        {
                            var fileSb = new StringBuilder();
                            fileSb.AppendLine($"--- Splat to Cell Mapping (Total: {mappedSplatCount} mapped splats out of {splatCount}) ---");
                            for (int i = 0; i < splatCount; i++)
                            {
                                uint cellId = GaussianSplatAsset.m_splatcell[i];
                                if (cellId != 0xFFFFFFFF)
                                {
                                    fileSb.AppendLine($"Splat[{i,7}]: Cell ID = {cellId}");
                                }
                            }
                            fileSb.AppendLine($"--- End of Mapping ---");
                            File.WriteAllText(path, fileSb.ToString(), Encoding.UTF8);
                            Debug.Log($"Successfully saved splat-cell mapping to {path}");
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"Failed to save splat-cell mapping to file: {e.Message}");
                        }
                    }
                    // 매핑에 사용된 고유한 (활성화된) 셀 인덱스들을 파일로 저장
                    string activeCellPath = EditorUtility.SaveFilePanel("Save Active Cell Indices", "", "active_cell_indices.txt", "txt");
                    if (!string.IsNullOrEmpty(activeCellPath))
                    {
                        try
                        {
                            var activeCells = new HashSet<uint>();
                            for (int i = 0; i < splatCount; i++)
                            {
                                uint cellId = GaussianSplatAsset.m_splatcell[i]; 
                                if (cellId != 0xFFFFFFFF)
                                {
                                    activeCells.Add(cellId);
                                }
                            }

                            using (var writer = new StreamWriter(activeCellPath, false, Encoding.UTF8))
                            {
                                writer.WriteLine($"--- Active Cell Indices (Total: {activeCells.Count}) ---");
                                var sortedCells = new List<uint>(activeCells);
                                sortedCells.Sort();
                                foreach (uint cellId in sortedCells)
                                    writer.WriteLine(cellId);
                                writer.WriteLine($"--- End of Indices ---");
                            }
                            Debug.Log($"Successfully saved active cell indices to {activeCellPath}");
                        }
                        catch (Exception e)
                        {
                            Debug.LogError($"Failed to save active cell indices to file: {e.Message}");
                        }
                    }

                    // 시각화를 위해 GPU로 데이터 업로드
                    /*gs.UploadSplatCellLinkData(GaussianSplatAsset.m_splatcell);*/
                }
            }
        }

        void MultiEditGUI()
        {
            DrawSeparator();
            CountTargetSplats(out var totalSplats, out var totalObjects);
            EditorGUILayout.LabelField("Total Objects", $"{totalObjects}");
            EditorGUILayout.LabelField("Total Splats", $"{totalSplats:N0}");
            if (totalSplats > GaussianSplatAsset.kMaxSplats)
            {
                EditorGUILayout.HelpBox($"Can't merge, too many splats (max. supported {GaussianSplatAsset.kMaxSplats:N0})", MessageType.Warning);
                return;
            }

            var targetGs = (GaussianSplatRenderer) target;
            if (!targetGs || !targetGs.HasValidAsset || !targetGs.isActiveAndEnabled)
            {
                EditorGUILayout.HelpBox($"Can't merge into {target.name} (no asset or disable)", MessageType.Warning);
                return;
            }

            if (targetGs.asset.chunkData != null)
            {
                EditorGUILayout.HelpBox($"Can't merge into {target.name} (needs to use Very High quality preset)", MessageType.Warning);
                return;
            }
            if (GUILayout.Button($"Merge into {target.name}"))
            {
                MergeSplatObjects();
            }
        }

        void CountTargetSplats(out int totalSplats, out int totalObjects)
        {
            totalObjects = 0;
            totalSplats = 0;
            foreach (var obj in targets)
            {
                var gs = obj as GaussianSplatRenderer;
                if (!gs || !gs.HasValidAsset || !gs.isActiveAndEnabled)
                    continue;
                ++totalObjects;
                totalSplats += gs.splatCount;
            }
        }

        void MergeSplatObjects()
        {
            CountTargetSplats(out var totalSplats, out _);
            if (totalSplats > GaussianSplatAsset.kMaxSplats)
                return;
            var targetGs = (GaussianSplatRenderer) target;

            int copyDstOffset = targetGs.splatCount;
            targetGs.EditSetSplatCount(totalSplats);
            foreach (var obj in targets)
            {
                var gs = obj as GaussianSplatRenderer;
                if (!gs || !gs.HasValidAsset || !gs.isActiveAndEnabled)
                    continue;
                if (gs == targetGs)
                    continue;
                gs.EditCopySplatsInto(targetGs, 0, copyDstOffset, gs.splatCount);
                copyDstOffset += gs.splatCount;
                gs.gameObject.SetActive(false);
            }
            Debug.Assert(copyDstOffset == totalSplats, $"Merge count mismatch, {copyDstOffset} vs {totalSplats}");
            Selection.activeObject = targetGs;
        }

        static EditorWindow GetMainGameView()
        {
            var gameViewType = Type.GetType("UnityEditor.GameView,UnityEditor");
            if (gameViewType == null)
            {
                Debug.LogError("Failed to find UnityEditor.GameView type. This might be due to a Unity version change.");
                return null;
            }

            // Try to find the main game view using the internal static method
            var getMainGameView = gameViewType.GetMethod("GetMainGameView", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            if (getMainGameView != null)
            {
                var res = getMainGameView.Invoke(null, null);
                if (res is EditorWindow window)
                    return window;
            }

            // Fallback: find any open game view window
            var gameViews = Resources.FindObjectsOfTypeAll(gameViewType);
            if (gameViews.Length > 0)
                return (EditorWindow)gameViews[0];

            Debug.LogError("Failed to get main game view. Please ensure the Game View window is open and visible.");
            return null;
        }
        
        static void ShowOriginalImages(GaussianSplatRenderer gs)
        {
            var asset = gs.asset;
            if (asset == null || asset.cameras == null || asset.cameras.Length == 0)
            {
                Debug.LogWarning("No cameras found in the Gaussian Splat asset.");
                return;
            }

            string imageFolder = EditorUtility.OpenFolderPanel("Select Folder with Original Images", "", "");
            if (string.IsNullOrEmpty(imageFolder))
                return;

            const string containerName = "Original Images";
            var container = gs.transform.Find(containerName);
            if (container != null)
            {
                DestroyImmediate(container.gameObject);
            }

            var containerGO = new GameObject(containerName);
            container = containerGO.transform;
            container.SetParent(gs.transform, false);

            var cameras = asset.cameras;
            for (int i = 0; i < cameras.Length; i++)
            {
                var camInfo = cameras[i];
                if (string.IsNullOrEmpty(camInfo.imageName))
                    continue;

                string imagePath = Path.Combine(imageFolder, camInfo.imageName);
                if (!File.Exists(imagePath))
                {
                    Debug.LogWarning($"Image not found for camera {i}: {imagePath}");
                    continue;
                }

                EditorUtility.DisplayProgressBar("Loading Original Images", $"Processing camera {i + 1}/{cameras.Length}", (float)i / cameras.Length);

                var quadGO = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quadGO.name = camInfo.imageName;
                quadGO.transform.SetParent(container, false);

                quadGO.transform.localPosition = camInfo.pos;
                quadGO.transform.localRotation = Quaternion.LookRotation(camInfo.axisZ, camInfo.axisY);

                float distance = 1.0f;

                byte[] fileData = File.ReadAllBytes(imagePath);
                Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                tex.LoadImage(fileData);

                float aspect = (float)tex.width / tex.height;
                float quadHeight = 2.0f * distance * Mathf.Tan(camInfo.fov * 0.5f * Mathf.Deg2Rad);
                quadGO.transform.localScale = new Vector3(quadHeight * aspect, quadHeight, 1);

                var quadRenderer = quadGO.GetComponent<Renderer>();
                var shader = Shader.Find("GaussianSplatting/UnlitDoubleSided");
                if (shader == null)
                {
                    Debug.LogWarning("Shader 'GaussianSplatting/UnlitDoubleSided' not found, falling back to 'Unlit/Texture'. Images may not be visible from behind.");
                    shader = Shader.Find("Unlit/Texture");
                }
                var mat = new Material(shader) { mainTexture = tex };
                quadRenderer.material = mat;

                DestroyImmediate(quadGO.GetComponent<Collider>());
            }
            EditorUtility.ClearProgressBar();
        }

        static void ShowSegmentedImages(GaussianSplatRenderer gs)
        {
            var asset = gs.asset;
            if (asset == null || asset.cameras == null || asset.cameras.Length == 0)
            {
                Debug.LogWarning("No cameras found in the Gaussian Splat asset.");
                return;
            }

            string imageFolder = EditorUtility.OpenFolderPanel("Select Folder with Original Images", "", "");
            if (string.IsNullOrEmpty(imageFolder))
                return;

            string segmentFolder = Path.Combine(imageFolder, "segment");
            Directory.CreateDirectory(segmentFolder);

            const string containerName = "Segmented Images";
            var container = gs.transform.Find(containerName);
            if (container != null)
            {
                DestroyImmediate(container.gameObject);
            }

            var containerGO = new GameObject(containerName);
            container = containerGO.transform;
            container.SetParent(gs.transform, false);

            int imagesFound = 0;

            var cameras = asset.cameras;
            for (int i = 0; i < cameras.Length; i++)
            {
                var camInfo = cameras[i];
                if (string.IsNullOrEmpty(camInfo.imageName))
                    continue;

                string imagePath = Path.Combine(segmentFolder, $"{Path.GetFileNameWithoutExtension(camInfo.imageName)}_seg.png");
                if (!File.Exists(imagePath))
                    continue;

                imagesFound++;
                EditorUtility.DisplayProgressBar("Loading Segmented Images", $"Processing camera {i + 1}/{cameras.Length}", (float)i / cameras.Length);

                var quadGO = GameObject.CreatePrimitive(PrimitiveType.Quad);
                quadGO.name = Path.GetFileName(imagePath);
                quadGO.transform.SetParent(container, false);

                quadGO.transform.localPosition = camInfo.pos;
                quadGO.transform.localRotation = Quaternion.LookRotation(camInfo.axisZ, camInfo.axisY);

                byte[] fileData = File.ReadAllBytes(imagePath);
                Texture2D tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
                tex.LoadImage(fileData);

                float aspect = (float)tex.width / tex.height;
                float quadHeight = 2.0f * 1.0f * Mathf.Tan(camInfo.fov * 0.5f * Mathf.Deg2Rad);
                quadGO.transform.localScale = new Vector3(quadHeight * aspect, quadHeight, 1);

                var quadRenderer = quadGO.GetComponent<Renderer>();
                var shader = Shader.Find("GaussianSplatting/UnlitDoubleSided") ?? Shader.Find("Unlit/Texture");
                var mat = new Material(shader) { mainTexture = tex };
                quadRenderer.material = mat;

                DestroyImmediate(quadGO.GetComponent<Collider>());
            }
            EditorUtility.ClearProgressBar();

            if (imagesFound == 0)
            {
                EditorUtility.DisplayDialog("No Segmented Images Found", $"Could not find any segmented images in the '{segmentFolder}' folder.\n\nPlease run 'Segment and Colorize' first to generate them.", "OK");
                DestroyImmediate(containerGO);
            }
        }

        [Serializable]
        private class SamColorResponse
        {
            public string image;
            public string error;
        }

        static async void SegmentAndColorizeWithSAM(GaussianSplatRenderer gs)
        {
            var asset = gs.asset;
            if (asset == null || asset.cameras == null || asset.cameras.Length == 0)
            {
                Debug.LogWarning("No cameras found in the Gaussian Splat asset.");
                return;
            }

            string imageFolder = EditorUtility.OpenFolderPanel("Select Folder with Original Images", "", "");
            if (string.IsNullOrEmpty(imageFolder))
                return;

            string outputFolder = Path.Combine(imageFolder, "segment");
            Directory.CreateDirectory(outputFolder);

            var cameras = asset.cameras;
            AssetDatabase.StartAssetEditing();
            try
            {
                for (int i = 0; i < cameras.Length; i++)
                {
                    var camInfo = cameras[i];
                    if (string.IsNullOrEmpty(camInfo.imageName))
                        continue;

                    string imagePath = Path.Combine(imageFolder, camInfo.imageName);
                    if (!File.Exists(imagePath))
                    {
                        Debug.LogWarning($"Image not found for camera {i}: {imagePath}");
                        continue;
                    }

                    string progressMessage = $"Segmenting & Coloring image {i + 1}/{cameras.Length}: {camInfo.imageName}";
                    if (EditorUtility.DisplayCancelableProgressBar("Segmenting with SAM", progressMessage, (float)i / cameras.Length))
                        break;

                    byte[] imageData = File.ReadAllBytes(imagePath);
                    var form = new List<IMultipartFormSection>
                    {
                        new MultipartFormFileSection("image", imageData, Path.GetFileName(imagePath), "image/jpeg")
                    };

                    string url = "http://127.0.0.1:5000/segment";
                    using var request = UnityWebRequest.Post(url, form);
                    var asyncOp = request.SendWebRequest();
                    while (!asyncOp.isDone)
                    {
                        await Task.Yield();
                    }

                    if (request.result != UnityWebRequest.Result.Success)
                    {
                        Debug.LogError($"Error sending request to SAM server: {request.error}. Make sure the server is running.");
                        if (!EditorUtility.DisplayDialog("Server Error", $"Failed to connect to the SAM server for {camInfo.imageName}. Make sure the server is running. Do you want to continue?", "Continue", "Abort"))
                            break;
                        continue;
                    }

                    string jsonResponse = request.downloadHandler.text;
                    var response = JsonUtility.FromJson<SamColorResponse>(jsonResponse);

                    if (!string.IsNullOrEmpty(response.error))
                    {
                        Debug.LogError($"SAM server returned an error for {camInfo.imageName}: {response.error}");
                        if (!EditorUtility.DisplayDialog("SAM Error", $"Segmentation failed for {camInfo.imageName}. See console for details. Do you want to continue?", "Continue", "Abort"))
                            break;
                    }
                    else if (!string.IsNullOrEmpty(response.image))
                    {
                        string outputImageName = $"{Path.GetFileNameWithoutExtension(camInfo.imageName)}_seg.png";
                        string outputPath = Path.Combine(outputFolder, outputImageName);
                        byte[] imageDataResult = Convert.FromBase64String(response.image);
                        File.WriteAllBytes(outputPath, imageDataResult);
                    }
                }
            }
            finally
            {
                AssetDatabase.StopAssetEditing();
                EditorUtility.ClearProgressBar();
                AssetDatabase.Refresh();
                Debug.Log("SAM segmentation process finished.");
            }
        }

        void EditGUI(GaussianSplatRenderer gs)
        {
            ++s_EditStatsUpdateCounter;

            DrawSeparator();
            bool wasToolActive = ToolManager.activeContextType == typeof(GaussianToolContext);
            GUILayout.BeginHorizontal();
            bool isToolActive = GUILayout.Toggle(wasToolActive, "Edit", EditorStyles.miniButton);
            using (new EditorGUI.DisabledScope(!gs.editModified))
            {
                if (GUILayout.Button("Reset", GUILayout.ExpandWidth(false)))
                {
                    if (EditorUtility.DisplayDialog("Reset Splat Modifications?",
                            $"This will reset edits of {gs.name} to match the {gs.asset.name} asset. Continue?",
                            "Yes, reset", "Cancel"))
                    {
                        gs.enabled = false;
                        gs.enabled = true;
                    }
                }
            }

            GUILayout.EndHorizontal();
            if (!wasToolActive && isToolActive)
            {
                ToolManager.SetActiveContext<GaussianToolContext>();
                if (Tools.current == Tool.View)
                    Tools.current = Tool.Move;
            }

            if (wasToolActive && !isToolActive)
            {
                ToolManager.SetActiveContext<GameObjectToolContext>();
            }

            if (isToolActive && gs.asset.chunkData != null)
            {
                EditorGUILayout.HelpBox("Splat move/rotate/scale tools need Very High splat quality preset", MessageType.Warning);
            }

            EditorGUILayout.Space();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Add Cutout"))
            {
                GaussianCutout cutout = ObjectFactory.CreateGameObject("GSCutout", typeof(GaussianCutout)).GetComponent<GaussianCutout>();
                Transform cutoutTr = cutout.transform;
                cutoutTr.SetParent(gs.transform, false);
                cutoutTr.localScale = (gs.asset.boundsMax - gs.asset.boundsMin) * 0.25f;
                gs.m_Cutouts ??= Array.Empty<GaussianCutout>();
                ArrayUtility.Add(ref gs.m_Cutouts, cutout);
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
                Selection.activeGameObject = cutout.gameObject;
            }
            if (GUILayout.Button("Use All Cutouts"))
            {
                gs.m_Cutouts = FindObjectsByType<GaussianCutout>(FindObjectsSortMode.InstanceID);
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
            }

            if (GUILayout.Button("No Cutouts"))
            {
                gs.m_Cutouts = Array.Empty<GaussianCutout>();
                gs.UpdateEditCountsAndBounds();
                EditorUtility.SetDirty(gs);
            }
            GUILayout.EndHorizontal();
            EditorGUILayout.PropertyField(m_PropCutouts);

            bool hasCutouts = gs.m_Cutouts != null && gs.m_Cutouts.Length != 0;
            bool modifiedOrHasCutouts = gs.editModified || hasCutouts;

            var asset = gs.asset;
            EditorGUILayout.Space();
            EditorGUI.BeginChangeCheck();
            m_ExportBakeTransform = EditorGUILayout.Toggle("Export in world space", m_ExportBakeTransform);
            if (EditorGUI.EndChangeCheck())
            {
                EditorPrefs.SetBool(kPrefExportBake, m_ExportBakeTransform);
            }

            if (GUILayout.Button("Export PLY"))
                ExportPlyFile(gs, m_ExportBakeTransform);
            if (asset.posFormat > GaussianSplatAsset.VectorFormat.Norm16 ||
                asset.scaleFormat > GaussianSplatAsset.VectorFormat.Norm16 ||
                asset.colorFormat > GaussianSplatAsset.ColorFormat.Float16x4 ||
                asset.shFormat > GaussianSplatAsset.SHFormat.Float16)
            {
                EditorGUILayout.HelpBox(
                    "It is recommended to use High or VeryHigh quality preset for editing splats, lower levels are lossy",
                    MessageType.Warning);
            }

            bool displayEditStats = isToolActive || modifiedOrHasCutouts;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Splats", $"{gs.splatCount:N0}");
            if (displayEditStats)
            {
                EditorGUILayout.LabelField("Cut", $"{gs.editCutSplats:N0}");
                EditorGUILayout.LabelField("Deleted", $"{gs.editDeletedSplats:N0}");
                EditorGUILayout.LabelField("Selected", $"{gs.editSelectedSplats:N0}");
                if (hasCutouts)
                {
                    if (s_EditStatsUpdateCounter > 10)
                    {
                        gs.UpdateEditCountsAndBounds();
                        s_EditStatsUpdateCounter = 0;
                    }
                }
            }
        }

        static void DrawSeparator()
        {
            EditorGUILayout.Space(12f, true);
            GUILayout.Box(GUIContent.none, "sv_iconselector_sep", GUILayout.Height(2), GUILayout.ExpandWidth(true));
            EditorGUILayout.Space();
        }

        bool HasFrameBounds()
        {
            return true;
        }

        Bounds OnGetFrameBounds()
        {
            var gs = target as GaussianSplatRenderer;
            if (!gs || !gs.HasValidRenderSetup)
                return new Bounds(Vector3.zero, Vector3.one);
            Bounds bounds = default;
            bounds.SetMinMax(gs.asset.boundsMin, gs.asset.boundsMax);
            if (gs.editSelectedSplats > 0)
            {
                bounds = gs.editSelectedBounds;
            }
            bounds.extents *= 0.7f;
            return TransformBounds(gs.transform, bounds);
        }

        public static Bounds TransformBounds(Transform tr, Bounds bounds )
        {
            var center = tr.TransformPoint(bounds.center);

            var ext = bounds.extents;
            var axisX = tr.TransformVector(ext.x, 0, 0);
            var axisY = tr.TransformVector(0, ext.y, 0);
            var axisZ = tr.TransformVector(0, 0, ext.z);

            // sum their absolute value to get the world extents
            ext.x = Mathf.Abs(axisX.x) + Mathf.Abs(axisY.x) + Mathf.Abs(axisZ.x);
            ext.y = Mathf.Abs(axisX.y) + Mathf.Abs(axisY.y) + Mathf.Abs(axisZ.y);
            ext.z = Mathf.Abs(axisX.z) + Mathf.Abs(axisY.z) + Mathf.Abs(axisZ.z);

            return new Bounds { center = center, extents = ext };
        }

        static unsafe void ExportPlyFile(GaussianSplatRenderer gs, bool bakeTransform)
        {
            var path = EditorUtility.SaveFilePanel(
                "Export Gaussian Splat PLY file", "", $"{gs.asset.name}-edit.ply", "ply");
            if (string.IsNullOrWhiteSpace(path))
                return;

            int kSplatSize = UnsafeUtility.SizeOf<Utils.InputSplatData>();
            using var gpuData = new GraphicsBuffer(GraphicsBuffer.Target.Structured, gs.splatCount, kSplatSize);

            if (!gs.EditExportData(gpuData, bakeTransform))
                return;

            Utils.InputSplatData[] data = new Utils.InputSplatData[gpuData.count];
            gpuData.GetData(data);

            var gpuDeleted = gs.GpuEditDeleted;
            uint[] deleted = new uint[gpuDeleted.count];
            gpuDeleted.GetData(deleted);

            // count non-deleted splats
            int aliveCount = 0;
            for (int i = 0; i < data.Length; ++i)
            {
                int wordIdx = i >> 5;
                int bitIdx = i & 31;
                bool isDeleted = (deleted[wordIdx] & (1u << bitIdx)) != 0;
                bool isCutout = data[i].nor.sqrMagnitude > 0;
                if (!isDeleted && !isCutout)
                    ++aliveCount;
            }

            using FileStream fs = new FileStream(path, FileMode.Create, FileAccess.Write);
            // note: this is a long string! but we don't use multiline literal because we want guaranteed LF line ending
            var header = $"ply\nformat binary_little_endian 1.0\nelement vertex {aliveCount}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\nproperty float f_rest_0\nproperty float f_rest_1\nproperty float f_rest_2\nproperty float f_rest_3\nproperty float f_rest_4\nproperty float f_rest_5\nproperty float f_rest_6\nproperty float f_rest_7\nproperty float f_rest_8\nproperty float f_rest_9\nproperty float f_rest_10\nproperty float f_rest_11\nproperty float f_rest_12\nproperty float f_rest_13\nproperty float f_rest_14\nproperty float f_rest_15\nproperty float f_rest_16\nproperty float f_rest_17\nproperty float f_rest_18\nproperty float f_rest_19\nproperty float f_rest_20\nproperty float f_rest_21\nproperty float f_rest_22\nproperty float f_rest_23\nproperty float f_rest_24\nproperty float f_rest_25\nproperty float f_rest_26\nproperty float f_rest_27\nproperty float f_rest_28\nproperty float f_rest_29\nproperty float f_rest_30\nproperty float f_rest_31\nproperty float f_rest_32\nproperty float f_rest_33\nproperty float f_rest_34\nproperty float f_rest_35\nproperty float f_rest_36\nproperty float f_rest_37\nproperty float f_rest_38\nproperty float f_rest_39\nproperty float f_rest_40\nproperty float f_rest_41\nproperty float f_rest_42\nproperty float f_rest_43\nproperty float f_rest_44\nproperty float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n";
            fs.Write(Encoding.UTF8.GetBytes(header));
            for (int i = 0; i < data.Length; ++i)
            {
                int wordIdx = i >> 5;
                int bitIdx = i & 31;
                bool isDeleted = (deleted[wordIdx] & (1u << bitIdx)) != 0;
                bool isCutout = data[i].nor.sqrMagnitude > 0;
                if (!isDeleted && !isCutout)
                {
                    var splat = data[i];
                    byte* ptr = (byte*)&splat;
                    fs.Write(new ReadOnlySpan<byte>(ptr, kSplatSize));
                }
            }

            Debug.Log($"Exported PLY {path} with {aliveCount:N0} splats");
        }
    }
}