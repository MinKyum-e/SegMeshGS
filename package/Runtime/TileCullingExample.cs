// SPDX-License-Identifier: MIT

using System.Collections.Generic;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Example demonstrating how to use tile-based culling for 3D Gaussian Splat segmentation.
    /// This class shows the intended usage pattern of the tile culling feature.
    /// </summary>
    public class TileCullingExample : MonoBehaviour
    {
        [SerializeField] private GaussianSplatRenderer m_SplatRenderer;
        [SerializeField] private Camera m_Camera;
        
        private List<int> m_SplatIndicesBuffer = new List<int>();
        
        void Start()
        {
            if (m_SplatRenderer == null)
                m_SplatRenderer = FindObjectOfType<GaussianSplatRenderer>();
            
            if (m_Camera == null)
                m_Camera = Camera.main;
        }
        
        void Update()
        {
            // Update tile culling data once per frame
            if (m_SplatRenderer != null && m_Camera != null)
            {
                m_SplatRenderer.UpdateTileCulling(m_Camera);
            }
        }
        
        /// <summary>
        /// Example of how to find the most influential splat for a specific pixel.
        /// This would be called during segmentation to efficiently identify
        /// which splats to consider for each pixel.
        /// </summary>
        /// <param name="screenX">Pixel X coordinate</param>
        /// <param name="screenY">Pixel Y coordinate</param>
        /// <returns>Index of most influential splat, or -1 if none found</returns>
        public int FindMostInfluentialSplatForPixel(int screenX, int screenY)
        {
            if (m_SplatRenderer == null || !m_SplatRenderer.HasValidTileCulling)
                return -1;
                
            // Get all splats that may influence this pixel's cell
            int splatCount = m_SplatRenderer.GetSplatsInfluencingCell(screenX, screenY, m_SplatIndicesBuffer);
            
            if (splatCount == 0)
                return -1;
            
            // In a real implementation, you would:
            // 1. Calculate each splat's alpha contribution at this pixel
            // 2. Apply front-to-back compositing order
            // 3. Return the splat with highest contribution
            
            // For this example, we just return the first splat
            return m_SplatIndicesBuffer[0];
        }
        
        /// <summary>
        /// Example of how to perform efficient pixel-wise segmentation.
        /// This demonstrates the performance benefit of tile-based culling.
        /// </summary>
        /// <param name="regionRect">Screen region to process</param>
        /// <returns>Array of splat indices, one per pixel in the region</returns>
        public int[] SegmentScreenRegion(Rect regionRect)
        {
            if (m_SplatRenderer == null || !m_SplatRenderer.HasValidTileCulling)
                return new int[0];
                
            int width = Mathf.RoundToInt(regionRect.width);
            int height = Mathf.RoundToInt(regionRect.height);
            int[] result = new int[width * height];
            
            int startX = Mathf.RoundToInt(regionRect.x);
            int startY = Mathf.RoundToInt(regionRect.y);
            
            // Process each pixel
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int screenX = startX + x;
                    int screenY = startY + y;
                    int pixelIndex = y * width + x;
                    
                    // Use tile culling to get candidate splats
                    int splatCount = m_SplatRenderer.GetSplatsInfluencingCell(screenX, screenY, m_SplatIndicesBuffer);
                    
                    if (splatCount > 0)
                    {
                        // In a real implementation, calculate the most influential splat
                        // considering alpha blending and depth order
                        result[pixelIndex] = FindMostInfluentialSplatForPixel(screenX, screenY);
                    }
                    else
                    {
                        result[pixelIndex] = -1; // No splat influences this pixel
                    }
                }
            }
            
            return result;
        }
    }
}