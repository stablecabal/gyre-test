import numpy as np

from .test_harness import ALGORITHMS, GenerationTestCase

(lambda: 1)()  # Stop items below here from sorting above it

from gyre.pipeline.prompt_types import Prompt, PromptFragment

WEIGHTS = {
    "moredaisies": 0.5,
    "moretulips": -0.5,
    "matched": 0,
}


class PromptsWeightTest(GenerationTestCase):
    enginecfg = "prompt_weights.engine.yaml"

    def _params(self):
        return {
            "height": 512,
            "width": 512,
            "guidance_scale": 7.5,
            "sampler": ALGORITHMS["k_euler"],
            "num_inference_steps": 25,
            "seed": 420420420,
        }

    def _prompt_fragments(self, i):
        return dict(
            init=[
                PromptFragment("A DSLR photo of ", 1.0),
            ],
            base=[
                PromptFragment("a meadow filled with ", 1.0),
                PromptFragment("daisies", 1.0 + i),
                PromptFragment(" and ", 1.0),
                PromptFragment("tulips", 1.0 - i),
                PromptFragment(", f/2.8 35mm Portra 400. ", 1.0),
            ],
            spacer=[
                PromptFragment(", ".join(["quality" * 60]) + ". ", 1.0),
            ],
            extension=[
                PromptFragment(
                    "A stormy day. Heavy rain, clouds, strong wind, raindrops on camera. ",
                    1.5,
                ),
            ],
        )

    def test_prompt_weights(self):
        for label, i in WEIGHTS.items():
            fragments = self._prompt_fragments(i)
            prompt = Prompt(fragments["init"] + fragments["base"])

            with self.with_engine() as engine:
                result = engine(prompt=prompt, **self._params())[0]
                self.assertImageSnapshotMatches(label, result)

    def test_long_prompt_weights(self):
        for label, i in WEIGHTS.items():
            fragments = self._prompt_fragments(i)
            prompt = Prompt(
                fragments["init"]
                + fragments["extension"]
                + fragments["spacer"]
                + fragments["base"]
            )

            with self.with_engine() as engine:
                result = engine(prompt=prompt, **self._params())[0]
                self.assertImageSnapshotMatches(label, result)

    def test_long_reversed_prompt_weights(self):
        for label, i in WEIGHTS.items():
            fragments = self._prompt_fragments(i)
            prompt = Prompt(
                fragments["init"]
                + fragments["base"]
                + fragments["spacer"]
                + fragments["extension"]
            )

            with self.with_engine() as engine:
                result = engine(prompt=prompt, **self._params())[0]
                self.assertImageSnapshotMatches(label, result)
