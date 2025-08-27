# UnitySegGS : Segmentation Object & 3D Gaussian Splatting tool in Unity
- Unity Version : **2022.3.21f1**

---

## Pipeline
<details>
  <summary>Click to expand</summary>

# 3D Reconstruction Pipeline

## 1. Video Input
- Input format: **`.mp4`**
- Load video file from user

---

## 2. Frame Extraction
- Tool: **FFmpeg**
- Extract frames from the video

---

## 3. Point Cloud Generation
- Tool: **COLMAP**
- Generate point cloud via SfM & MVS

---

## 4. 3D Gaussian Splatting (3DGS)

### 4-1. Standard 3DGS Generation
- Create **3DGS model (splat)** from COLMAP results
- Tool: **sugar**  
  - Run until just before mesh extraction

---

### 4-2. Object-Specific 3DGS Generation
1. **SAM & CLIP**  
   - Object recognition and masking in images
   - Select Object (text)
3. **Mask Image Generation**  
   - Generate mask for the selected object
4. **COLMAP (Feature Extraction mode)**  
   - Use `--ImageReader.mask_path` option  
   - Extract features and reconstruct using masked images
5. **sugar (with mesh extraction)**  
   - Generate object-focused 3DGS and mesh

---

## 5. Mesh Extraction Feature
- Provide a separate **Mesh Extraction button**  
- Allow mesh extraction from splats generated in **4-1**

</details>

---

## References
<details>
  <summary>Click to expand</summary>

- **3DGS Viewer**  
  https://github.com/aras-p/UnityGaussianSplatting  

- **Splat to Mesh (SuGaR)**  
  https://github.com/Anttwo/SuGaR  

</details>
