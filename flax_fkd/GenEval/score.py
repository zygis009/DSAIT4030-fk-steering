import argparse
import json
import os
import re
import sys
import time

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc
zsc.tqdm = lambda it, *args, **kwargs: it
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
COLOR_CLASSIFIERS = {}


class ImageCrops(torch.utils.data.Dataset):
    def __init__(self, image: Image.Image, objects, transform):
        self._image = image.convert("RGB")
        bgcolor = "#999"
        if bgcolor == "original":
            self._blank = self._image.copy()
        else:
            self._blank = Image.new("RGB", image.size, color=bgcolor)
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape), (index, self._image.size[::-1], mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        
        image = image.crop(box[:4])
        return (self._transform(image), 0)

class GenEval:
    def __init__(self, mask2former_path: str = None, options: any = None):
        self.options = options if options is not None else {}

        self._load_models(checkpoint_path=mask2former_path)

        self.threshold = float(self.options.get('threshold', 0.3))
        self.counting_threshold = float(self.options.get('counting_threshold', 0.9))
        self.max_objects = int(self.options.get('max_objects', 16))
        self.nms_threshold = float(self.options.get('max_overlap', 1.0))
        self.position_threshold = float(self.options.get('position_threshold', 0.1))

    def _load_models(self, checkpoint_path: str = None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 
                "../../models/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
            )
        config_path = os.path.join(
            os.path.dirname(mmdet.__file__),
            "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
        self.object_detector = init_detector(config_path, checkpoint_path, device=DEVICE)

        clip_arch = "ViT-L-14"
        self.clip_model, _, self.transform = open_clip.create_model_and_transforms(clip_arch, pretrained="openai", device=DEVICE)
        self.tokenizer = open_clip.get_tokenizer(clip_arch)

        with open(os.path.join(os.path.dirname(__file__), "prompts/object_names.txt")) as cls_file:
            self.classnames = [line.strip() for line in cls_file]


    def color_classification(self, image: Image.Image, bboxes: list, classname: str):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                self.clip_model, self.tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                DEVICE
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes, self.transform),
            batch_size=16, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, DEVICE)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(self, box_a, box_b):
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(self, obj_a, obj_b):
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(np.abs(offset) - self.position_threshold * (dim_a + dim_b), 0) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5: relations.add("left of")
        if dx > 0.5: relations.add("right of")
        if dy < -0.5: relations.add("above")
        if dy > 0.5: relations.add("below")
        return relations

    def evaluate(self, image: Image.Image, objects: object , metadata: object):
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = objects.get(classname, [])[:req['count']]
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                if 'color' in req:
                    # Color check
                    colors = self.color_classification(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found " +
                            f"{colors.count(req['color'])} {req['color']}; and " +
                            ", ".join(f"{colors.count(c)} {c}" for c in COLORS if c in colors)
                        )
                if 'position' in req and matched:
                    # Relative position check
                    expected_rel, target_group = req['position']
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self.relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found " +
                                        f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(objects.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(objects[classname])}")
        return correct, "\n".join(reason)

    def evaluate_one(self, image_name: str, image: Image.Image, metadata: object):
        result = inference_detector(self.object_detector, np.array(image.convert("RGB")))
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        image = ImageOps.exif_transpose(image)
        detected = {}
        # Determine bounding boxes to keep
        confidence_threshold = self.threshold if metadata['tag'] != "counting" else self.counting_threshold
        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold] # Threshold
            ordering = ordering[:self.max_objects].tolist() # Limit number of detected objects per class
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((bbox[index][max_obj], None if segm is None else segm[index][max_obj]))
                ordering = [
                    obj for obj in ordering
                    if self.nms_threshold == 1 or self.compute_iou(bbox[index][max_obj], bbox[index][obj]) < self.nms_threshold
                ]
            if not detected[classname]:
                del detected[classname]
        # Evaluate
        is_correct, reason = self.evaluate(image, detected, metadata)
        return {
            'filename': image_name,
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
            'correct': is_correct,
            'reason': reason,
            'metadata': json.dumps(metadata),
            'details': json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        }
    
    def evaluate_many(self, image_names: list[str], images: list[Image.Image], metadatas: list[object] | str, output_path: str = None):
        if isinstance(metadatas, str):
            with open(metadatas, "r") as f:
                metadatas = [json.loads(line) for line in f]
        
        full_results = pd.DataFrame([self.evaluate_one(image_name, image, metadata) for (image_name, image, metadata) in zip(image_names, images, metadatas)])
        # Save results if needed
        if output_path is not None:
            if os.path.dirname(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as fp:
                full_results.to_json(fp, orient="records", lines=True)
        # Compute and return score
        task_scores = []
        for tag, tasks in full_results.groupby('tag', sort=False):
            task_scores.append(tasks['correct'].mean())
            print(f"{tag:<16} = {tasks['correct'].mean():.2%} ({tasks['correct'].sum()} / {len(tasks)})")
        
        return np.mean(task_scores)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-path", type=str, default=None)
    # Other arguments
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)

    return args

if __name__ == "__main__":
    args = parse_args()

    if DEVICE == "cpu":
        print("Warning: CUDA is not available. Performance may be severely impacted.", file=sys.stderr)

    # Instantiate the evaluator
    genEval = GenEval(
        mask2former_path=args.model_path,
        options=args.options
    )

    image_names = [os.path.join(args.imagedir, image_name) for image_name in os.listdir(args.imagedir) if os.path.splitext(image_name)[1] in ['.jpg', '.png']]
    images = [Image.open(image_path) for image_path in image_names]
    metadatas = [{"tag": "color_attr", "include": [{"class": "knife", "count": 1, "color": "brown"}, {"class": "donut", "count": 1, "color": "blue"}], "prompt": "a photo of a brown knife and a blue donut"}] * len(image_names)

    # Run the full evaluation process
    score = genEval.evaluate_many(image_names=image_names, images=images, metadatas=metadatas, output_path=args.outfile)
    print(f"Evaluation completed. Results saved to {args.outfile}; Computed score: {score}", file=sys.stderr)