// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;
using Seg3dgsTool.Runtime;

namespace Seg3dgsTool.Editor
{
    [CustomEditor(typeof(PreprocessingTools))]
    public class PreprocessingToolsEditor  : UnityEditor.Editor
    {
        SerializedProperty m_PropVideoPath;

        public void OnEnable()
        {
            m_PropVideoPath = serializedObject.FindProperty("m_VideoPath");
        }

        public override void OnInspectorGUI()
        {
            var tool = (PreprocessingTools)target;

            serializedObject.Update();

            EditorGUILayout.LabelField("Preprocessing tools", EditorStyles.boldLabel);

            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(m_PropVideoPath, new GUIContent("Video Path"));
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space();
            
            if (GUILayout.Button("Select Video File (.mp4)"))
            {
                string path = EditorUtility.OpenFilePanel("Select a Video", "", "mp4");
                if (!string.IsNullOrEmpty(path))
                {
                    m_PropVideoPath.stringValue = path;
                    Debug.Log("Selected video path: " + path);
                }
            }
            if (GUILayout.Button("Extract Frames with FFmpeg"))
            {
                tool.ExtractFramesFromVideo();
            }
            using (new EditorGUI.DisabledScope(tool.IsRunning || string.IsNullOrEmpty(m_PropVideoPath.stringValue)))
            {

                if (GUILayout.Button("Run COLMAP Conversion"))
                {
                    tool.RunColmapConversion();
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
