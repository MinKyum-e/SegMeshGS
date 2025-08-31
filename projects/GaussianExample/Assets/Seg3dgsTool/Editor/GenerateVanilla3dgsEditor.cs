// SPDX-License-Identifier: MIT

using UnityEditor;
using UnityEngine;
using Seg3dgsTool.Runtime;

namespace Seg3dgsTool.Editor
{
    [CustomEditor(typeof(GenerateVanilla3dgs))]
    public class GenerateVanilla3dgsEditor : UnityEditor.Editor
    {
        SerializedProperty m_ColmapPath;

        public void OnEnable()
        {
            m_ColmapPath = serializedObject.FindProperty("m_ColmapPath");
        }

        public override void OnInspectorGUI()
        {
            var tool = (GenerateVanilla3dgs)target;

            serializedObject.Update();

            EditorGUILayout.LabelField("Generate Vanilla 3dgs", EditorStyles.boldLabel);

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
                
                if (GUILayout.Button("Generate Vanilla 3dgs"))
                {
                    tool.RunWslScriptViaServer();
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
