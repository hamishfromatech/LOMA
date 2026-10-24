"""The images field on a chat turn.

Ollama wants bare base64; a browser gives you a data URL. The API is the seam
where that is normalised and where an oversized or malformed upload is refused
before a model spends twenty seconds on it.
"""

from __future__ import annotations

from server.images import MAX_IMAGES, read_images

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class TestReadImages:
    def test_absent_field_means_no_images(self):
        assert read_images({}) == ([], None)

    def test_a_data_url_is_stripped_to_bare_base64(self):
        images, problem = read_images({"images": [f"data:image/png;base64,{PNG}"]})
        assert problem is None
        assert images == [PNG]

    def test_bare_base64_passes_through(self):
        images, problem = read_images({"images": [PNG]})
        assert problem is None and images == [PNG]

    def test_whitespace_and_newlines_are_removed(self):
        wrapped = PNG[:20] + "\n" + PNG[20:]
        images, problem = read_images({"images": [wrapped]})
        assert problem is None and images == [PNG]

    def test_too_many_images_is_refused(self):
        images, problem = read_images({"images": [PNG] * (MAX_IMAGES + 1)})
        assert images == []
        assert problem and "at most" in problem

    def test_a_non_string_is_refused(self):
        _, problem = read_images({"images": [123]})
        assert problem and "base64" in problem

    def test_an_empty_string_is_refused(self):
        _, problem = read_images({"images": [""]})
        assert problem

    def test_non_list_is_refused(self):
        _, problem = read_images({"images": "a.png"})
        assert problem and "list" in problem

    def test_an_oversized_image_is_refused_without_decoding_it(self):
        huge = "A" * 12_000_000  # ~9 MB decoded, over the 8 MB cap
        _, problem = read_images({"images": [huge]})
        assert problem and "larger than" in problem
