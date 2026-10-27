import torch
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


class TableStructureDetector:

    LABEL_MAP = {
        0: 'table',
        1: 'column',
        2: 'row',
        3: 'column_header',
        4: 'projected_row_header',
        5: 'spanning_cell'
    }

    def __init__(
        self,
        model_name: str = "microsoft/table-transformer-structure-recognition",
        device: str = "cuda:0",
        confidence_threshold: float = 0.7
    ):
        from transformers import DetrImageProcessor, TableTransformerForObjectDetection

        print(f"Loading Table Transformer: {model_name}")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.confidence_threshold = confidence_threshold
        print("Table Transformer loaded.")

    def detect(self, image: Image.Image) -> Dict[str, List[BBox]]:

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=self.confidence_threshold, target_sizes=target_sizes
        )[0]

        structure = {
            'rows': [],
            'columns': [],
            'headers': [],
            'spanning_cells': [],
            'table': None
        }

        for score, label, box in zip(
            results['scores'].cpu().tolist(),
            results['labels'].cpu().tolist(),
            results['boxes'].cpu().tolist()
        ):
            x1, y1, x2, y2 = box
            label_name = self.LABEL_MAP.get(label, 'unknown')
            bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label_name, confidence=score)

            if label_name == 'row':
                structure['rows'].append(bbox)
            elif label_name == 'column':
                structure['columns'].append(bbox)
            elif label_name in ('column_header', 'projected_row_header'):
                structure['headers'].append(bbox)
            elif label_name == 'spanning_cell':
                structure['spanning_cells'].append(bbox)
            elif label_name == 'table':
                structure['table'] = bbox

        structure['rows'].sort(key=lambda b: b.y1)
        structure['columns'].sort(key=lambda b: b.x1)
        structure['headers'].sort(key=lambda b: b.y1)

        return structure

    def detect_from_path(self, image_path: str) -> Dict[str, List[BBox]]:
        image = Image.open(image_path).convert('RGB')
        return self.detect(image)


def get_row_patches(
    image: Image.Image,
    structure: Dict[str, List[BBox]],
    padding: int = 10,
    min_patch_height: int = 32
) -> List[Tuple[Image.Image, BBox]]:

    w, h = image.size
    patches = []

    for row_bbox in structure['rows']:
        x1 = max(0, int(row_bbox.x1) - padding)
        y1 = max(0, int(row_bbox.y1) - padding)
        x2 = min(w, int(row_bbox.x2) + padding)
        y2 = min(h, int(row_bbox.y2) + padding)

        if (y2 - y1) < min_patch_height:
            continue

        patch = image.crop((x1, y1, x2, y2))
        adjusted_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                             label='row', confidence=row_bbox.confidence)
        patches.append((patch, adjusted_bbox))

    return patches


def get_header_patch(
    image: Image.Image,
    structure: Dict[str, List[BBox]],
    padding: int = 10
) -> Optional[Tuple[Image.Image, BBox]]:

    w, h = image.size

    if structure['headers']:
        header = structure['headers'][0]
        x1 = max(0, int(header.x1) - padding)
        y1 = max(0, int(header.y1) - padding)
        x2 = min(w, int(header.x2) + padding)
        y2 = min(h, int(header.y2) + padding)
    elif structure['rows']:
        first_row = structure['rows'][0]
        x1 = max(0, int(first_row.x1) - padding)
        y1 = max(0, int(first_row.y1) - padding)
        x2 = min(w, int(first_row.x2) + padding)
        y2 = min(h, int(first_row.y2) + padding)
    else:
        return None

    patch = image.crop((x1, y1, x2, y2))
    bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2, label='header', confidence=1.0)
    return (patch, bbox)


def get_column_patches(
    image: Image.Image,
    structure: Dict[str, List[BBox]],
    padding: int = 10
) -> List[Tuple[Image.Image, BBox]]:

    w, h = image.size
    patches = []

    for col_bbox in structure['columns']:
        x1 = max(0, int(col_bbox.x1) - padding)
        y1 = max(0, int(col_bbox.y1) - padding)
        x2 = min(w, int(col_bbox.x2) + padding)
        y2 = min(h, int(col_bbox.y2) + padding)

        patch = image.crop((x1, y1, x2, y2))
        adjusted_bbox = BBox(x1=x1, y1=y1, x2=x2, y2=y2,
                             label='column', confidence=col_bbox.confidence)
        patches.append((patch, adjusted_bbox))

    return patches


def get_full_table_patch(
    image: Image.Image,
    max_size: int = 1024
) -> Tuple[Image.Image, BBox]:

    w, h = image.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
    else:
        resized = image.copy()

    bbox = BBox(x1=0, y1=0, x2=w, y2=h, label='full_table', confidence=1.0)
    return (resized, bbox)


def visualize_structure(
    image: Image.Image,
    structure: Dict[str, List[BBox]],
    output_path: str = None
) -> Image.Image:

    from PIL import ImageDraw, ImageFont

    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    color_map = {
        'row': 'red',
        'column': 'blue',
        'column_header': 'green',
        'projected_row_header': 'green',
        'header': 'green',
        'spanning_cell': 'orange',
        'table': 'purple'
    }

    for category in ['rows', 'columns', 'headers', 'spanning_cells']:
        elements = structure.get(category, [])
        for i, bbox in enumerate(elements):
            color = color_map.get(bbox.label, 'gray')
            draw.rectangle(bbox.as_tuple(), outline=color, width=2)
            draw.text(
                (bbox.x1 + 3, bbox.y1 + 2),
                f"{bbox.label}_{i} ({bbox.confidence:.2f})",
                fill=color
            )

    if output_path:
        vis.save(output_path)

    return vis
