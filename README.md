# UnitySegMeshGS: Segmentation for 3D Gaussian Splatting in Unity
- **Unity Version:** 2022.3.21f1

---

## Pipeline
<details>
  <summary>Click to expand</summary>

# 3D Reconstruction Pipeline

## 1. Video Input
- Supported format: **`.mp4`**
- User uploads a video file

---

## 2. Frame Extraction
- Tool: **FFmpeg**
- Extract frames from the uploaded video

---

## 3. Point Cloud Generation
- Tool: **COLMAP**
- Generate a point cloud using Structure-from-Motion (SfM) and Multi-View Stereo (MVS)

---

## 4. 3D Gaussian Splatting (3DGS)

### 4-1. Segmentation Tool (SAGA)
- Convert COLMAP results into a **3DGS model** for segmentation using **SAGA**
- Steps:
  - Follow the SAGA pipeline to generate a splat (.ply) containing marking data derived from SAM images
  - SAGA provides a viewer where users can manually select specific objects for segmentation
  - After completion, the segmented objects are automatically imported back into Unity
    
### 4-2. Mesh Extraction Tool (SuGaR)
- Convert COLMAP results into a **3DGS model** using **SuGaR**
- Steps:
  - Run SuGaR until just before mesh extraction
  - If a mesh file (.obj) is needed, execute the mesh extraction process

---



</details>

---

## References
<details>
  <summary>Click to expand</summary>

- **3DGS Viewer**  
  [https://github.com/aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)  

- **Splat to Mesh (SuGaR)**  
  [https://github.com/Anttwo/SuGaR](https://github.com/Anttwo/SuGaR)  

- **Splat Segmentation (SAGA)**  
  [https://github.com/Jumpat/SegAnyGAussians](https://github.com/Jumpat/SegAnyGAussians)  

</details>
