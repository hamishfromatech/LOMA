"""What a turn may carry as an image.

Flask-free on purpose: this is validation, and validation should be testable
without a web framework installed. The browser hands over data URLs because that
is what a FileReader produces; Ollama wants the bare base64 payload.

    { "images": ["data:image/png;base64,iVBOR..."] }
"""

from __future__ import annotations

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def read_images(payload: dict) -> tuple[list[str], str | None]:
    """Validate the ``images`` field: base64 only, at most four, size-capped.

    The UI sends data URLs ("data:image/png;base64,…") because that is what a
    FileReader gives you, and Ollama wants the bare base64 payload.
    """
    raw = payload.get("images")
    if raw is None:
        return [], None
    if not isinstance(raw, list):
        return [], "images must be a list of base64 strings"
    if len(raw) > MAX_IMAGES:
        return [], f"at most {MAX_IMAGES} images per turn"

    cleaned: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item:
            return [], f"image {index} is not a base64 string"
        value = item.strip()
        if value.startswith("data:"):
            _, _, value = value.partition(",")
        value = "".join(value.split())
        if not value:
            return [], f"image {index} is empty"
        # 4/3 is the base64 expansion; this catches an oversized upload before
        # it reaches the model rather than after a 20-second wait.
        if len(value) * 3 / 4 > MAX_IMAGE_BYTES:
            return [], f"image {index} is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB"
        cleaned.append(value)
    return cleaned, None
