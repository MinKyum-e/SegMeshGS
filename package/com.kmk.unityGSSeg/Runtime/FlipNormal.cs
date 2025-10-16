
using UnityEngine;

namespace SegNeshGS.Runtime
{
    [RequireComponent(typeof(MeshFilter))]
    public class FlipNormal : MonoBehaviour
    {
        public void Flip()
        {
            MeshFilter mf = GetComponent<MeshFilter>();
            if (mf == null)
            {
                Debug.LogError("not found MeshFilter ");
                return;
            }

            Mesh mesh = mf.sharedMesh;
            if (mesh == null)
            {
                Debug.LogError("not found Mesh");
                return;
            }
            Vector3[] normals = mesh.normals;
            for (int i = 0; i < normals.Length; i++)
            {
                normals[i] = -normals[i];
            }

            mesh.normals = normals;
            for (int subMesh = 0; subMesh < mesh.subMeshCount; subMesh++)
            {
                int[] triangles = mesh.GetTriangles(subMesh);
                for (int i = 0; i < triangles.Length; i += 3)
                {
                    int temp = triangles[i];
                    triangles[i] = triangles[i + 1];
                    triangles[i + 1] = temp;
                }

                mesh.SetTriangles(triangles, subMesh);
            }

            mesh.RecalculateBounds();

            Debug.Log(gameObject.name + " mesh flipped");
        }
    }
}