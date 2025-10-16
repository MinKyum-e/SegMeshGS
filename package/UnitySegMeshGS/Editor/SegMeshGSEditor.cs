// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;
using SegNeshGS.Runtime;


namespace SegNeshGS.Editor
{
    [CustomEditor(typeof(SegMeshGS))]
    public class SegMeshGSEditor :UnityEditor.Editor
    {
        SerializedProperty m_PropVideoPath;
        SerializedProperty m_Query;
        public void OnEnable()
        {
            m_PropVideoPath = serializedObject.FindProperty("m_VideoPath");
            m_Query = serializedObject.FindProperty("m_Query");
        }

        public override void OnInspectorGUI()
        {
            var tool = (SegMeshGS)target;

            serializedObject.Update();

            EditorGUILayout.LabelField("SegMeshGS Full Pipeline", EditorStyles.boldLabel);

            
            // Video Path (Read-only)
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(m_PropVideoPath, new GUIContent("Video Path"));
            EditorGUI.EndDisabledGroup();
            
            EditorGUILayout.Space();
            
            using (new EditorGUI.DisabledScope(tool.IsRunning))
            {
                if (GUILayout.Button("Select Video File (.mp4)"))
                {
                    string path = EditorUtility.OpenFilePanel("Select a Video", "", "mp4");
                    if (!string.IsNullOrEmpty(path))
                    {
                        m_PropVideoPath.stringValue = path;
                        Debug.Log("Selected video path: " + path);
                    }
                }
                
                EditorGUILayout.Space();
                if (GUILayout.Button("Run SegMeshGS Full Pipeline", GUILayout.Height(35)))
                {
                    tool.RunSegMeshGSFullPipeline();
                }
            }

            if (tool.IsRunning)
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox($"Running...\n{tool.CurrentStatus}", MessageType.Info);
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}