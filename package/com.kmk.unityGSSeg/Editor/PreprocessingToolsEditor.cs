// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;
using SegNeshGS.Runtime;

namespace SegNeshGS.Editor
{
    [CustomEditor(typeof(PreprocessingTools))]
    public class PreprocessingToolsEditor : UnityEditor.Editor
    {
        SerializedProperty m_PropVideoPath;
        bool showAdvanced = false;

        public void OnEnable()
        {
            m_PropVideoPath = serializedObject.FindProperty("m_VideoPath");
        }

        public override void OnInspectorGUI()
        {
            var tool = (PreprocessingTools)target;

            serializedObject.Update();

            EditorGUILayout.LabelField("Preprocessing tools", EditorStyles.boldLabel);

            // Video Path (Read-only)
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(m_PropVideoPath, new GUIContent("Video Path"));
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space();

            using (new EditorGUI.DisabledScope(tool.IsRunning ))
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
                
                showAdvanced = EditorGUILayout.Foldout(showAdvanced, "For Debug");
                if (showAdvanced)
                {
                    EditorGUI.indentLevel++;

                    if (GUILayout.Button("Extract Frames with FFmpeg"))
                    {
                        tool.ExtractFramesFromVideo();
                    }
                    if (GUILayout.Button("Run COLMAP Conversion"))
                    {
                        tool.RunColmapConversion();
                    }
                    EditorGUI.indentLevel--;
                }

                EditorGUILayout.Space();
                if (GUILayout.Button("Run COLMAP Full Pipeline", GUILayout.Height(35)))
                {
                    tool.RunColmapFullPipeline();
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
