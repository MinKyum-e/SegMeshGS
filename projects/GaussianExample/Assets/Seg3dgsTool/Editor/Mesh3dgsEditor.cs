using Seg3dgsTool.Runtime;
using UnityEditor;
using UnityEngine;

namespace Seg3dgsTool.Editor
{
    [CustomEditor(typeof(Mesh3dgs))]
    public class Mesh3dgsEditor : UnityEditor.Editor
    {
        SerializedProperty m_ColmapPath;
        SerializedProperty m_Query;

        public void OnEnable()
        {
            m_ColmapPath = serializedObject.FindProperty("m_ColmapPath");
            m_Query = serializedObject.FindProperty("m_Query");
        }

        public override void OnInspectorGUI()
        {
            var tool = (Mesh3dgs)target;

            serializedObject.Update();

            EditorGUILayout.LabelField("ClipSAM Tools", EditorStyles.boldLabel);
            EditorGUILayout.Space();

            // COLMAP Path (Read-only)
            EditorGUI.BeginDisabledGroup(true);
            EditorGUILayout.PropertyField(m_ColmapPath, new GUIContent("COLMAP Project Path"));
            EditorGUI.EndDisabledGroup();

            if (GUILayout.Button("Select COLMAP Project Folder"))
            {
                string path = EditorUtility.OpenFolderPanel("Select COLMAP Project Folder", "", "");
                if (!string.IsNullOrEmpty(path))
                {
                    m_ColmapPath.stringValue = path;
                }
            }

            EditorGUILayout.Space();

            EditorGUILayout.PropertyField(m_Query, new GUIContent("Query Text"));

            EditorGUILayout.Space(10);

            using (new EditorGUI.DisabledScope(tool.IsRunning || string.IsNullOrEmpty(m_ColmapPath.stringValue) || string.IsNullOrEmpty(m_Query.stringValue)))
            {
                if (GUILayout.Button("Run ClipSAM"))
                {
                    tool.RunClipSam(false);
                }

                if (GUILayout.Button("Run Fast-ClipSAM"))
                {
                    tool.RunClipSam(true);
                }
                
                EditorGUILayout.Space(10);
                if (GUILayout.Button("Extract Mesh", GUILayout.Height(35)))
                {
                    tool.RunExtractMesh();
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