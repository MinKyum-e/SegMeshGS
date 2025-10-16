// SPDX-License-Identifier: MIT

using SegNeshGS.Runtime;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace SegNeshGS.Editor
{
    [CustomEditor(typeof(Show3DGS))]
    public class Show3DGSEditor : UnityEditor.Editor
    {
        SerializedProperty m_PropPlyPath;

        private void OnEnable()
        {
            m_PropPlyPath = serializedObject.FindProperty("m_PlyPath");
        }

        public override void OnInspectorGUI()
        {
            var tool = (Show3DGS)target;

            serializedObject.Update();

            GUILayout.Label("Create 3DGS Instance from PLY", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            EditorGUILayout.PropertyField(m_PropPlyPath, new GUIContent("PLY File Path"));

            if (GUILayout.Button("Select .ply File"))
            {
                string path = EditorUtility.OpenFilePanel("Select .ply file", "", "ply");
                if (!string.IsNullOrEmpty(path))
                {
                    m_PropPlyPath.stringValue = path;
                }
            }

            EditorGUILayout.Space();

            using (new EditorGUI.DisabledScope(string.IsNullOrEmpty(m_PropPlyPath.stringValue)))
            {
                if (GUILayout.Button("Generate 3DGS Instance"))
                {
                    Make3DGSInstance(tool.m_PlyPath);
                }
            }

            serializedObject.ApplyModifiedProperties();
        }

        private void Make3DGSInstance(string plyPath)
        {
            if (string.IsNullOrEmpty(plyPath) || !File.Exists(plyPath))
            {
                Debug.LogError($"PLY file not found or path is invalid: {plyPath}");
                return;
            }

            string outputAssetFolder = "Assets/GaussianAssets";
            Debug.Log($"Creating GaussianSplatAsset from '{plyPath}' into '{outputAssetFolder}'...");
            
            // 에디터 스크립트에서 에디터 전용 클래스를 호출합니다.
            GaussianSplatAsset gsAsset = GaussianSplatAssetCreator.CreateAssetFromPath(plyPath, outputAssetFolder);

            if (gsAsset == null)
            {
                Debug.LogError($"Failed to create GaussianSplatAsset from PLY file: {plyPath}");
                return;
            }

            GameObject gsPrefab = Resources.Load<GameObject>("GaussianSplats");

            if (gsPrefab == null)
            {
                Debug.LogError("Failed to load 'GaussianSplats.prefab' from Resources folder. Make sure it exists.");
                return;
            }

            GameObject gsInstance = (GameObject)PrefabUtility.InstantiatePrefab(gsPrefab);
            gsInstance.name = $"GaussianSplat_{Path.GetFileNameWithoutExtension(plyPath)}";

            GaussianSplatRenderer gsRenderer = gsInstance.GetComponent<GaussianSplatRenderer>();
            if (gsRenderer == null)
            {
                Debug.LogError($"The 'GaussianSplats' prefab does not have a {nameof(GaussianSplatRenderer)} component.");
                DestroyImmediate(gsInstance); 
                return;
            }

            gsRenderer.m_Asset = gsAsset;
            
            Selection.activeGameObject = gsInstance;
            Debug.Log($"Successfully created Gaussian Splat instance for: {plyPath}");
        }
    }
}