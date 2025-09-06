// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;
using Seg3dgsTool.Runtime;

namespace Seg3dgsTool.Editor
{
    [CustomEditor(typeof(Segment3dgs))]
    public class Segment3dgsEditor : UnityEditor.Editor
    {
        SerializedProperty m_ColmapPath;
        SerializedProperty m_Downsample;

        public void OnEnable()
        {
            m_ColmapPath = serializedObject.FindProperty("m_ColmapPath");
            m_Downsample = serializedObject.FindProperty("m_Downsample");
        }

        public override void OnInspectorGUI()
        {
            var tool = (Segment3dgs)target;

            serializedObject.Update();

           

            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(m_ColmapPath, new GUIContent("COLMAP path"));
            EditorGUI.EndDisabledGroup();

            EditorGUILayout.Space();

            using (new EditorGUI.DisabledScope(tool.IsRunning))
            {
                if (GUILayout.Button("Select COLMAP Project Folder"))
                {
                    string path = EditorUtility.OpenFolderPanel("Select COLMAP Project Folder", "", "");
                    if (!string.IsNullOrEmpty(path))
                    {
                        m_ColmapPath.stringValue = path;
                        Debug.Log("Selected COLMAP folder path: " + path);
                    }
                }
                
                EditorGUILayout.Space(20);
                

                
                EditorGUILayout.LabelField("Generate Vanilla 3dgs", EditorStyles.boldLabel);
                EditorGUILayout.Space(2);
                if (GUILayout.Button("Generate Vanilla 3dgs"))
                {
                    tool.RunWslScriptViaServer();
                }
                
                EditorGUILayout.Space(20);
                EditorGUILayout.LabelField("Extract SAM Mask", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(m_Downsample, new GUIContent("Downsample Factor"));
                EditorGUILayout.Space(2);
                if (GUILayout.Button("Run SAM Mask Extraction"))
                {
                    tool.RunSamExtractionViaServer();
                }
                
                
                EditorGUILayout.Space(20);
                EditorGUILayout.LabelField("Get corresponding mask scales", EditorStyles.boldLabel);
                EditorGUILayout.Space(2);
                if (GUILayout.Button("Run corresponding mask scales"))
                {
                    tool.RunGetScaleViaServer();
                }

            }

            if (tool.IsRunning)
            {
                EditorGUILayout.Space();
                EditorGUILayout.HelpBox($"Running...\n{tool.CurrentStatus}", MessageType.Info);
                Repaint();
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
