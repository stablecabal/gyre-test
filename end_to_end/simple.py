import json
import unittest

import requests

from tools.constants import BASE_HTTP
from tools.images import prep_image
from tools.utils import snapshot_test


class EndToEndSimpleRESTAPITest(unittest.TestCase):
    def _filter_list_engine(self, data):
        data["engines"] = [
            engine
            for engine in data["engines"]
            if engine.get("id") in {"stable-diffusion-v1-5", "hat-gan-x4"}
        ]
        return data

    @snapshot_test(filter=lambda self, data: self._filter_list_engine(data))
    def test_list_engines(self):
        return requests.get(f"{BASE_HTTP}/v1/engines/list")

    @snapshot_test
    def test_txt2img(self):
        return requests.post(
            f"{BASE_HTTP}/v1/generation/stable-diffusion-v1-5/text-to-image",
            headers={"Content-Type": "application/json", "Accept": "image/png"},
            data=json.dumps({"text_prompts": [{"text": "A Teddybear"}], "seed": 12345}),
        )

    @snapshot_test
    def test_img2img(self):
        return requests.post(
            f"{BASE_HTTP}/v1/generation/stable-diffusion-v1-5/image-to-image",
            headers={"Accept": "image/png"},
            data={
                "text_prompts[0][text]": "A Teddybear",
                "seed": 12345,
                "image_strength": 0.7,
            },
            files={
                "init_image": ("init_image.png", prep_image("rabbit")[1], "image/png")
            },
        )

    @snapshot_test
    def test_inpaint(self):
        return requests.post(
            f"{BASE_HTTP}/v1/generation/stable-diffusion-v1-5/image-to-image/masking",
            headers={"Accept": "image/png"},
            data={
                "text_prompts[0][text]": "A Teddybear",
                "text_prompts[1][text]": "vase, flowers",  # Need a negative prompt or you just get flowers
                "text_prompts[1][weight]": "-1.0",
                "seed": 12347,
                "mask_source": "INIT_IMAGE_ALPHA",
            },
            files={
                "init_image": (
                    "init_image.png",
                    prep_image("kitchen", crop_offset=1, mask="tabletop")[1],
                    "image/png",
                )
            },
        )

    @snapshot_test
    def test_upscale(self):
        return requests.post(
            f"{BASE_HTTP}/v1/generation/hat-gan-x4/image-to-image/upscale",
            headers={"Accept": "image/png"},
            data={
                "width": 2048,
            },
            files={"image": ("image.png", prep_image("garland")[1], "image/png")},
        )


if __name__ == "__main__":
    unittest.main()
