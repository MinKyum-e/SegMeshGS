# Tile-Based Culling for 3D Gaussian Splat Segmentation

## Overview

This feature implements tile-based culling to efficiently find splats that influence specific screen cells. This is particularly useful for 3D Gaussian Splatting segmentation where you need to identify which splat contributes most to each pixel's final color.

## Problem Solved

In 3D Gaussian Splatting, determining the most influential splat for a pixel requires:
1. Checking all splats that may affect the pixel
2. Calculating alpha contributions in depth order
3. Performing alpha compositing to find the dominant splat

Without culling, this means checking thousands or millions of splats per pixel, which is highly inefficient.

## Solution: Tile-Based Culling

The screen is divided into tiles (16x16 pixels each), and for each tile we precompute which splats may influence any pixel within that tile. This allows:
- **Spatial Culling**: Only consider spatially relevant splats
- **Reduced Computation**: Skip splats that definitely don't contribute
- **Cache Coherency**: Adjacent pixels share similar splat lists

## API Usage

### Basic Setup

```csharp
// Get reference to GaussianSplatRenderer
GaussianSplatRenderer splatRenderer = GetComponent<GaussianSplatRenderer>();
Camera camera = Camera.main;

// Update tile culling data (call once per frame or when camera changes)
splatRenderer.UpdateTileCulling(camera);
```

### Finding Splats for a Pixel

```csharp
List<int> splatIndices = new List<int>();
int screenX = 100, screenY = 200;

// Get all splats that may influence this pixel
int count = splatRenderer.GetSplatsInfluencingCell(screenX, screenY, splatIndices);

// Now process only these candidate splats instead of all splats
for (int i = 0; i < count; i++)
{
    int splatIndex = splatIndices[i];
    // Calculate this splat's contribution to the pixel
    // Apply alpha compositing, etc.
}
```

### Efficient Segmentation

```csharp
// Process a screen region efficiently
public int[] SegmentRegion(Rect region)
{
    int width = (int)region.width;
    int height = (int)region.height;
    int[] result = new int[width * height];
    
    List<int> candidateSplats = new List<int>();
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int screenX = (int)region.x + x;
            int screenY = (int)region.y + y;
            
            // Use tile culling to get candidate splats
            int count = splatRenderer.GetSplatsInfluencingCell(screenX, screenY, candidateSplats);
            
            // Find most influential splat among candidates
            int mostInfluential = FindMostInfluentialSplat(screenX, screenY, candidateSplats);
            result[y * width + x] = mostInfluential;
        }
    }
    
    return result;
}
```

## Performance Considerations

### GPU-CPU Synchronization
- `GetSplatsInfluencingCell()` performs GPU-to-CPU memory transfer
- Use sparingly - consider batching or GPU-only processing when possible
- Call `UpdateTileCulling()` only when camera parameters change

### Memory Usage
- Tile grid: `(screenWidth/16) * (screenHeight/16) * 4 bytes`
- Splat indices: `estimatedSplatTileReferences * 4 bytes`
- For 1920x1080: ~32KB for tile data + splat reference data

### Tile Size
- Currently fixed at 16x16 pixels to match existing Morton code tiles
- Smaller tiles = more precise culling but higher memory overhead
- Larger tiles = less memory but more false positives

## Integration with Existing Code

The tile culling system integrates with the existing rendering pipeline:
- Uses existing Morton encoding functions
- Leverages existing compute shader infrastructure  
- Reuses splat transformation and covariance calculations
- Compatible with existing edit/delete functionality

## Example Usage

See `TileCullingExample.cs` for a complete example showing:
- How to set up tile culling
- Basic pixel segmentation
- Region-based processing
- Performance optimization patterns

## Limitations

1. **Fixed tile size**: Currently hardcoded to 16x16 pixels
2. **Simple data structure**: Uses fixed-size arrays per tile (max 64 splats/tile)
3. **Conservative culling**: May include splats that don't actually contribute
4. **Synchronous readback**: `GetSplatsInfluencingCell()` blocks CPU

## Future Improvements

1. **Dynamic tile sizes** based on splat density
2. **Variable-length tile data** structures
3. **GPU-only processing** to avoid CPU-GPU synchronization
4. **Hierarchical culling** for very dense scenes
5. **Temporal coherence** to reuse data across frames