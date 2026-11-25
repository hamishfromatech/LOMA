"""What a turn may carry as an image.

Flask-free on purpose: this is validation, and validation should be testable
without a web framework installed. The browser hands over data URLs because that
is what a FileReader produces; Ollama wants the bare base64 payload.

    { "images": ["data:image/png;base64,iVBOR..."] }
"""

from __future__ import annotations

import base64

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024

#: The largest chat request body that could ever be legitimate. Images arrive as
#: base64, which is 4 characters per 3 bytes, so four 8 MB images are 42.7 MB of
#: JSON — and Flask's MAX_CONTENT_LENGTH was 32 MB. The documented limit was
#: unreachable: the request died in Flask before this validator ran, with an HTML
#: 413 the UI could not read. Exported so the app derives its cap from here and
#: the two cannot drift apart again.
MAX_BODY_BYTES = MAX_IMAGES * ((MAX_IMAGE_BYTES + 2) // 3 * 4) + 1024 * 1024


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
        # And it has to actually be base64. Anything else reaches Ollama and
        # comes back as an opaque 400 from a model call that already cost a
        # cold load, which reads to the user as "the model is broken".
        try:
            decoded = base64.b64decode(value, validate=True)
        except ValueError:
            return [], f"image {index} is not valid base64"
        if not decoded:
            return [], f"image {index} decoded to nothing"
        cleaned.append(value)
    return cleaned, None
