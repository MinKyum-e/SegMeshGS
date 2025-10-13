import os
import argparse
import torch
import clip
import numpy as np
from PIL import Image, ImageChops
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model, preprocess = clip.load("ViT-B/32", device=device)

def convert_box_xywh_to_xyxy(box):
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)

    # 크기 맞추기
    if segmentation_mask.shape != image_array.shape[:2]:
        segmentation_mask = cv2.resize(segmentation_mask.astype(np.uint8),
                                       (image_array.shape[1], image_array.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)

    segmented_array = np.zeros_like(image_array)
    segmented_array[segmentation_mask] = image_array[segmentation_mask]
    return Image.fromarray(segmented_array)

@torch.no_grad()
def retriev(elements, search_text: str):
    preprocessed_images = [preprocess(img).to(device) for img in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = clip_model.encode_image(stacked_images)
    text_features = clip_model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100. * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def process_folder(parent_folder, query, threshold):
    image_folder = os.path.join(parent_folder, "images")
    mask_folder = os.path.join(parent_folder, "sam_masks")
    output_folder = os.path.join(parent_folder, "seg_images", query)
    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(image_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_folder, filename)
        mask_path = os.path.join(mask_folder, f"{os.path.splitext(filename)[0]}.pt")

        if not os.path.exists(mask_path):
            print(f"Mask file missing: {mask_path}")
            continue

        print(f"Processing {filename} ...")

        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv).convert("RGB")

        masks = torch.load(mask_path, map_location="cpu") 
        if isinstance(masks, torch.Tensor):
            if masks.dim() == 3:
                masks = [m for m in masks] 
            elif masks.dim() == 2:
                masks = [masks]
            else:
                raise ValueError(f"Unsupported mask tensor shape: {masks.shape}")
        elif isinstance(masks, list):
            pass
        else:
            raise ValueError(f"Unsupported mask format: {type(masks)}")
        cropped_boxes = [segment_image(image_pil, m.numpy()) for m in masks]

        scores = retriev(cropped_boxes, query)
        indices = get_indices_of_values_above_threshold(scores, threshold)

        if not indices:
            print("No matching masks found. Skipping.")
            continue

        w, h = image_pil.size
        combined_mask = Image.new("L", (w, h), 0)
        for idx in indices:
            mask_img = Image.fromarray((masks[idx].numpy().astype(np.uint8) * 255)).resize((w, h), Image.NEAREST).convert("L")
            combined_mask = ImageChops.lighter(combined_mask, mask_img)


        white_bg = Image.new("RGB", (w, h), (255, 255, 255))
        masked_image = Image.composite(image_pil, white_bg, combined_mask)


        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        masked_image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP + precomputed SAM masks")
    parser.add_argument("--input_folder", type=str, required=True, help="Parent folder containing images/ and sam_masks/")
    parser.add_argument("--query", type=str, required=True, help="Text query for CLIP")
    parser.add_argument("--threshold", type=float, default=0.05, help="Score threshold for mask selection")
    args = parser.parse_args()

    process_folder(args.input_folder, args.query, args.threshold)
    print("All done!")
