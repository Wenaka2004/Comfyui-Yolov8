import folder_paths
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torch
import os
import ast

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class Yolov8DetectionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
            },
        }

    RETURN_TYPES = ("IMAGE", "JSON")
    FUNCTION = "detect"
    CATEGORY = "yolov8"

    def detect(self, image, model_name):
        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        
        print(f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        results = model(image)

        im_array = results[0].plot()
        im = Image.fromarray(im_array[...,::-1])

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        return (image_tensor_out, {"classify": [r.boxes.cls.tolist()[0] for r in results]})

class Yolov8SegNode:
    def __init__(self) -> None:
        ...
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",), 
                "model_name": (folder_paths.get_filename_list("yolov8"), ),
                "class_ids_text": ("STRING", {"default": "[0]"})
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "seg"
    CATEGORY = "yolov8"

    def seg(self, image, model_name, class_ids_text):
        try:
            class_ids = ast.literal_eval(class_ids_text)
            if not isinstance(class_ids, list):
                raise ValueError("Input must be a list of integers.")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input for class_ids: {class_ids_text}. Must be a list of integers, e.g., '[0, 1, 2]'. Error: {e}")

        image_tensor = image
        image_np = image_tensor.cpu().numpy()
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))
        
        print(f'model_path: {os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{model_name}')
        results = model(image)

        if results[0].masks is None:
            print("No masks found, returning empty mask.")
            combined_mask = torch.zeros(image_tensor.shape[2:], dtype=torch.int) * 255
        else:
            masks = results[0].masks.data
            boxes = results[0].boxes.data
            clss = boxes[:, 5]
            combined_mask = torch.zeros_like(masks[0], dtype=torch.int)
            for class_id in class_ids:
                indices = torch.where(clss == class_id)
                if indices[0].numel() > 0:
                    combined_mask = combined_mask | torch.any(masks[indices], dim=0).int()

            combined_mask = combined_mask * 255

        im_array = results[0].plot()
        im = Image.fromarray(im_array[...,::-1])

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)
        image_tensor_out = torch.unsqueeze(image_tensor_out, 0)

        return (image_tensor_out, combined_mask)

NODE_CLASS_MAPPINGS = {
    "Yolov8Detection": Yolov8DetectionNode,
    "Yolov8Segmentation": Yolov8SegNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yolov8Detection": "detection",
    "Yolov8Segmentation": "seg",
}
