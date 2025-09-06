using GaussianSplatting;
using UnityEngine;
using System.IO;
using GaussianSplatting.Runtime;
using GaussianSplatting.Editor;

namespace Seg3dgsTool.Runtime
{
    public class Show3DGS : MonoBehaviour
    {
        public string m_PlyPath;
        public int make3DGSInstance()
        {
            if (string.IsNullOrEmpty(m_PlyPath) || !File.Exists(m_PlyPath))
            {
                Debug.LogError($"PLY file not found or path is invalid: {m_PlyPath}");
                return Common.FAIL;
            }

            string outputAssetFolder = "Assets/GaussianAssets";
            Debug.Log($"Creating GaussianSplatAsset from '{m_PlyPath}' into '{outputAssetFolder}'...");
            GaussianSplatAsset gsAsset = GaussianSplatAssetCreator.CreateAssetFromPath(m_PlyPath, outputAssetFolder);

            if (gsAsset == null)
            {
                Debug.LogError($"Failed to create GaussianSplatAsset from PLY file: {m_PlyPath}");
                return Common.FAIL;
            }

            GameObject gsPrefab = Resources.Load<GameObject>("GaussianSplats");

            if (gsPrefab == null)
            {
                Debug.LogError("Failed to load 'GaussianSplats.prefab' from Resources folder. Make sure it exists.");
                return Common.FAIL;
            }

            GameObject gsInstance = Object.Instantiate(gsPrefab);
            gsInstance.name = $"GaussianSplat_{Path.GetFileNameWithoutExtension(m_PlyPath)}";

            GaussianSplatRenderer gsRenderer = gsInstance.GetComponent<GaussianSplatRenderer>();
            if (gsRenderer == null)
            {
                Debug.LogError($"The 'GaussianSplats' prefab does not have a {nameof(GaussianSplatRenderer)} component.");
                Object.Destroy(gsInstance); 
                return Common.FAIL;
            }

            gsRenderer.m_Asset = gsAsset;

            Debug.Log($"Successfully created Gaussian Splat instance for: {m_PlyPath}");

            return Common.SUCCESS;
        }
    }
}