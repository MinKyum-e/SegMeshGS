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
        SerializedProperty m_Iterations;
        SerializedProperty m_NumSampledRays;

        public void OnEnable()
        {
            m_ColmapPath = serializedObject.FindProperty("m_ColmapPath");
            m_Downsample = serializedObject.FindProperty("m_Downsample");
            m_Iterations = serializedObject.FindProperty("m_Iterations");
            m_NumSampledRays = serializedObject.FindProperty("m_NumSampledRays");
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

                EditorGUILayout.Space(20);
                EditorGUILayout.LabelField("Train Contrastive Feature", EditorStyles.boldLabel);
                EditorGUILayout.PropertyField(m_Iterations, new GUIContent("Iterations"));
                EditorGUILayout.PropertyField(m_NumSampledRays, new GUIContent("Num Sampled Rays"));
                EditorGUILayout.Space(2);
                if (GUILayout.Button("Run Contrastive Training"))
                {
                    tool.RunContrastiveTrainingViaServer();
                }
                
                EditorGUILayout.Space(20);
                EditorGUILayout.LabelField("SAGA GUI", EditorStyles.boldLabel); EditorGUILayout.Space(2);
                if (GUILayout.Button("Open SAGA GUI"))
                {
                    tool.RunSagaGuiViaServer();
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
