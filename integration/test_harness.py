import inspect
import os
import random
import unittest
from pathlib import Path

import grpc
import torch

# This line adds the various other module paths into the import searchpath
from gyre.generated import inject_generated_path
from gyre.src import inject_src_paths

inject_generated_path()
inject_src_paths()

import generation_pb2

from gyre import cache, engines_yaml, images
from gyre.manager import EngineManager, EngineMode
from gyre.ram_monitor import RamMonitor
from gyre.resources import ResourceProvider
from gyre.services.generate import GenerationServiceServicer
from tools.images import ssim

ALGORITHMS = {
    "ddim": generation_pb2.SAMPLER_DDIM,
    "plms": generation_pb2.SAMPLER_DDPM,
    "k_euler": generation_pb2.SAMPLER_K_EULER,
    "k_euler_ancestral": generation_pb2.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation_pb2.SAMPLER_K_HEUN,
    "k_dpm_2": generation_pb2.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation_pb2.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation_pb2.SAMPLER_K_LMS,
    "dpm_fast": generation_pb2.SAMPLER_DPM_FAST,
    "dpm_adaptive": generation_pb2.SAMPLER_DPM_ADAPTIVE,
    "dpmspp_1": generation_pb2.SAMPLER_DPMSOLVERPP_1ORDER,
    "dpmspp_2": generation_pb2.SAMPLER_DPMSOLVERPP_2ORDER,
    "dpmspp_3": generation_pb2.SAMPLER_DPMSOLVERPP_3ORDER,
    "dpmspp_2s_ancestral": generation_pb2.SAMPLER_K_DPMPP_2S_ANCESTRAL,
    "dpmspp_sde": generation_pb2.SAMPLER_K_DPMPP_SDE,
    "dpmspp_2m": generation_pb2.SAMPLER_K_DPMPP_2M,
}


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def add_callback(self, callback):
        pass

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details

    def abort(self, code, message):
        if code == grpc.StatusCode.OK:
            raise ValueError("Abort called with OK as status code")

        self.set_code(code)
        self.set_details(message)
        raise grpc.RpcError()


class GenerationTestCase(unittest.TestCase):
    enginecfg: str | None = None
    nsfw_behaviour = "ignore"
    vram_optimisation_level = 3

    @classmethod
    def setUpClass(cls):
        cls.ram_monitor = RamMonitor()
        cls.ram_monitor.start()

        folder = Path(__file__).parent

        engines, sources = engines_yaml.load([str(folder / cls.enginecfg)], {})

        cls.manager = EngineManager(
            engines,
            refresh_models=None,
            refresh_on_error=False,
            mode=EngineMode(vram_optimisation_level=cls.vram_optimisation_level),
            nsfw_behaviour=cls.nsfw_behaviour,
            ram_monitor=cls.ram_monitor,
        )

        cls.manager.loadPipelines()

        tensor_cache = cache.TensorLRUCache_Mem(512 * cache.MB)
        resource_provider = ResourceProvider(cache=tensor_cache.keyspace("resources:"))

        cls.generation_servicer = GenerationServiceServicer(
            cls.manager,
            tensor_cache=tensor_cache.keyspace("generation:"),
            resource_provider=resource_provider,
            supress_metadata=False,
            ram_monitor=cls.ram_monitor,
        )

    @classmethod
    def tearDownClass(cls):
        cls.ram_monitor.stop()

    def with_engine(self, id="testengine"):
        return self.manager.with_engine(id)

    def call_generator(self, request):
        self.context = FakeContext()
        return self.generation_servicer.Generate(request, self.context)

    def string_to_seed(self, string):
        return random.Random(string).randint(0, 2**32 - 1)

    def _flatten_outputs(self, output):
        if isinstance(output, list) or inspect.isgenerator(output):
            for item in output:
                yield from self._flatten_outputs(item)

        elif isinstance(output, torch.Tensor):
            if len(output.shape) == 4 and output.shape[0] > 1:
                yield from output.chunk(output.shape[0], dim=0)
            else:
                yield output

        elif isinstance(output, generation_pb2.Answer):
            yield from self._flatten_outputs(
                [
                    artifact
                    for artifact in output.artifacts
                    if artifact.type == generation_pb2.ARTIFACT_IMAGE
                ]
            )

        else:
            yield output

    def assertImageSnapshotMatches(self, prefix, results, ssim_threshold=0.05):
        source_path = Path(inspect.getfile(self.__class__)).parent

        class_name = self.__class__.__name__
        caller_name = inspect.currentframe().f_back.f_code.co_name

        name = f"{class_name}.{caller_name}"
        if prefix:
            name = name + "." + prefix

        snapshots_path = source_path / "snapshots"
        results_path = source_path / "results"

        for i, result in enumerate(list(self._flatten_outputs(results))):

            # Save result

            if isinstance(result, torch.Tensor):
                result = images.toWebpBytes(result)[0]
                result_path = results_path / f"{name}.{i}.webp"
                snapshot_outpath = snapshots_path / f"{name}.{i}.webp"
            elif isinstance(result, bytes):
                if result[0:3] == b"PNG":
                    result_path = results_path / f"{name}.{i}.png"
                    snapshot_outpath = snapshots_path / f"{name}.{i}.png"
                elif result[8:12] == b"WEBP":
                    result_path = results_path / f"{name}.{i}.webp"
                    snapshot_outpath = snapshots_path / f"{name}.{i}.webp"
                else:
                    self.assertTrue(False, "Can't identify image type for byestring")
            else:
                self.assertTrue(
                    False, f"Don't know how to handle type {type(result).__name__}"
                )

            result_path.parent.mkdir(exist_ok=True)
            result_path.write_bytes(result)

            # Update snapshot

            if os.environ.get("UPDATE_SNAPSHOT", False):
                snapshot_outpath.parent.mkdir(exist_ok=True)
                snapshot_outpath.write_bytes(result)

            # Compare to snapshot

            try:
                snapshot_path = next(snapshots_path.glob(f"{name}.{i}.*"))
            except StopIteration:
                snapshot_path = None

            if not (snapshot_path and snapshot_path.exists()):
                self.assertTrue(False, f"No snapshot found for {name}.{i}")

            self.assertLessEqual(ssim(result_path, snapshot_path), ssim_threshold)
