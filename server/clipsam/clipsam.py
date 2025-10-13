import os
import argparse
import cv2
import torch
import clip
import numpy as np
from PIL import Image, ImageChops
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_type = "vit_h"
sam_checkpoint = "/home/minkyum/SegAnyGAussians/third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

clip_model, preprocess = clip.load("ViT-B/32", device=device)


def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_array = np.zeros_like(image_array)
    segmented_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_array)
    return segmented_image

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


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(input_folder, filename)
        print(f"Processing {image_path} ...")

        # 이미지 읽기
        image_cv = cv2.imread(image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image_cv)

        image_pil = Image.open(image_path).convert("RGB")
        cropped_boxes = []
        for mask in masks:
            cropped_boxes.append(
                segment_image(image_pil, mask["segmentation"]).crop(
                    convert_box_xywh_to_xyxy(mask["bbox"])
                )
            )


        scores = retriev(cropped_boxes, query)
        indices = get_indices_of_values_above_threshold(scores, 0.05)

        if not indices:
            print("No matching masks found. Skipping.")
            continue

        w, h = image_pil.size
        combined_mask = Image.new("L", (w, h), 0)
        for idx in indices:
            mask_img = Image.fromarray(
                (masks[idx]["segmentation"].astype(np.uint8) * 255)
            ).convert("L")
            mask_img = mask_img.resize((w, h))
            combined_mask = ImageChops.lighter(combined_mask, mask_img)

        # 배경을 흰색으로 채움
        white_bg = Image.new("RGB", (w, h), (255, 255, 255))
        masked_image = Image.composite(image_pil, white_bg, combined_mask)

        # PNG로 저장
        output_path = os.path.join(
            output_folder, os.path.splitext(filename)[0] + ".png"
        )
        masked_image.save(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM + CLIP mask processing with white background")
    parser.add_argument("--input_folder", type=str, help="Path to input images folder")
    parser.add_argument("--query", type=str, required=True, help="Text query for CLIP matching")
    args = parser.parse_args()
    

    input_folder = os.path.join(args.input_folder, "images")
    if not os.path.isdir(input_folder):
        raise ValueError(f"'images' folder not found inside {input_folder}")
    query =args.query
    
    parent_folder = os.path.abspath(os.path.join(input_folder, os.pardir))
    output_folder = os.path.join(parent_folder, "seg_images", query)  # input 폴더와 같은 레벨
    process_folder(input_folder, output_folder)
    print("All done!")
