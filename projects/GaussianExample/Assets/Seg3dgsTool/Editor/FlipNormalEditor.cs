using UnityEngine;
using UnityEditor;
using Seg3dgsTool.Runtime;
namespace Seg3dgsTool.Editor
{
    [CustomEditor(typeof(FlipNormal))]
    public class FlipNormalEditor : UnityEditor.Editor

    {
        public override void OnInspectorGUI()
        {
            base.OnInspectorGUI();

            FlipNormal flipScript = (FlipNormal)target;

            if (GUILayout.Button("Flip Normals"))
            {
                flipScript.Flip();
            }
        }
    }
}