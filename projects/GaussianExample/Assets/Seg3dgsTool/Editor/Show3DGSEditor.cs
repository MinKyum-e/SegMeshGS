// SPDX-License-Identifier: MIT

using Seg3dgsTool.Runtime;
using UnityEditor;
using UnityEngine;

namespace Seg3dgsTool.Editor
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
                    tool.make3DGSInstance();
                }
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}