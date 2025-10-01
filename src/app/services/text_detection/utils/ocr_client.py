# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
OCR Client adapted for GCP.

Replaces Azure Form Recognizer with Google Cloud Vision OCR.
Keeps same public API: `ocr_client.read_text(image_stream)` returning
a list of (text, bounding_box) pairs where bounding_box is 4 (x,y) pixel coords.
"""

import io
import logging
from typing import List, Tuple, Union

from logger_config import get_logger

logger = get_logger(__name__)

# Try to import Google Vision
try:
    from google.cloud import vision
    from google.api_core.exceptions import GoogleAPIError
except ImportError as e:
    vision = None
    GoogleAPIError = Exception
    logger.error(
        "google-cloud-vision is not installed. "
        "Install it with: pip install google-cloud-vision"
    )


class OCRClient:
    def __init__(self):
        if vision is None:
            raise RuntimeError(
                "google-cloud-vision is required but not installed. "
                "Install with: pip install google-cloud-vision"
            )

        # Uses Application Default Credentials (GOOGLE_APPLICATION_CREDENTIALS or GCP runtime IAM)
        self.client = vision.ImageAnnotatorClient()

    def read_text(self, image_stream: Union[bytes, io.BytesIO]) -> List[Tuple[str, List[Tuple[int, int]]]]:
        """
        Run OCR on an image using Google Vision API.

        Args:
            image_stream: Bytes or file-like object containing the image.

        Returns:
            List of (text, bounding_box) pairs.
            bounding_box = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        if isinstance(image_stream, io.BytesIO):
            content = image_stream.getvalue()
        elif isinstance(image_stream, (bytes, bytearray)):
            content = image_stream
        else:
            raise ValueError("image_stream must be bytes or BytesIO")

        image = vision.Image(content=content)

        try:
            response = self.client.text_detection(image=image)
        except GoogleAPIError as e:
            logger.error(f"Vision API error: {e}")
            raise

        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")

        results: List[Tuple[str, List[Tuple[int, int]]]] = []

        annotations = response.text_annotations
        if not annotations:
            return results

        # Skip the first (full-text annotation)
        for ann in annotations[1:]:
            text = ann.description or ""
            vertices = []
            if ann.bounding_poly and ann.bounding_poly.vertices:
                for v in ann.bounding_poly.vertices:
                    vx = int(v.x) if v.x is not None else 0
                    vy = int(v.y) if v.y is not None else 0
                    vertices.append((vx, vy))

            # Normalize bbox to 4 points
            if len(vertices) >= 4:
                bbox = vertices[:4]
            elif len(vertices) > 0:
                xs = [p[0] for p in vertices]
                ys = [p[1] for p in vertices]
                bbox = [(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))]
            else:
                bbox = [(0, 0), (0, 0), (0, 0), (0, 0)]

            results.append((text, bbox))

        return results


# Default singleton client
ocr_client = OCRClient()
