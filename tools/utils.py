import functools
import inspect
import json
import unittest
from pathlib import Path

import requests

from .images import ssim


def check_requests_response(
    self: unittest.TestCase,
    res,
    snapshot_path: Path,
    result_path: Path,
    ssim_threshold,
    filter_res,
):
    self.assertEqual(
        res.status_code, 200, f"Request failed, {res.status_code}: {res.text}"
    )

    content_type = res.headers.get("Content-Type")
    if content_type == "image/png":
        snapshot_path = snapshot_path.with_suffix(".png")
        result_path = result_path.with_suffix(".png")

        with open(str(result_path), "wb") as fd:
            for chunk in res.iter_content(chunk_size=128):
                fd.write(chunk)

        self.assertTrue(
            snapshot_path.exists(), f"No snapshot available at {snapshot_path}"
        )

        ssim_res = ssim(snapshot_path, result_path)
        self.assertLess(ssim_res, ssim_threshold, "SSIM threshold exceeded")

    elif content_type == "application/json":
        snapshot_path = snapshot_path.with_suffix(".json")
        result_path = result_path.with_suffix(".json")

        # Read, parse, filter, and finally save (for later reference) the result
        result = res.json()
        if filter_res is not None:
            result = filter_res(self, result)
        result_path.write_text(json.dumps(result))

        # Read the snapshot json
        self.assertTrue(
            snapshot_path.exists(), f"No snapshot available at {snapshot_path}"
        )
        snapshot = json.loads(snapshot_path.read_text())

        # TODO: Process any images out of json

        result_path.write_text(json.dumps(result))
        self.assertEqual(json.dumps(result), json.dumps(snapshot))

    else:
        raise RuntimeError(f"Unknown response type {content_type}")


def snapshot_test(*args, **kwargs):
    if not args or not callable(args[0]):
        return functools.partial(snapshot_test, **kwargs)

    wrapped = args[0]
    ssim_threshold = kwargs.get("ssim_threshold", 0.05)
    filter_res = kwargs.get("filter", None)

    @functools.wraps(wrapped)
    def wrapper(self, *args, **kwargs):
        res = wrapped(self, *args, **kwargs)

        source_path = Path(inspect.getfile(wrapped))
        name = source_path.stem + "_" + wrapped.__name__
        base_path = source_path.parent

        snapshot_path = base_path / "snapshots" / name
        result_path = base_path / "results" / name
        result_path.parent.mkdir(exist_ok=True)

        if isinstance(res, requests.Response):
            check_requests_response(
                self, res, snapshot_path, result_path, ssim_threshold, filter_res
            )
        else:
            raise RuntimeError(
                f"Don't know how to compare result of type {type(res).__name__}"
            )

    return wrapper
