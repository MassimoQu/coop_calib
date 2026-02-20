import unittest
from unittest import mock

from opencood.tools import train_utils


class _DummyCfg(object):
    def __init__(self, enabled):
        self.enabled = bool(enabled)


class TrainUtilsPoseProviderCacheTest(unittest.TestCase):
    def test_maybe_apply_pose_provider_caches_config_on_same_hypes(self):
        hypes = {
            "pose_provider": {
                "enabled": True,
                "runtime_mode": "fusion_only",
                "pose_source": "noisy_input",
            }
        }
        batch = {"ego": {}}
        cfg_obj = _DummyCfg(enabled=True)

        with mock.patch.object(
            train_utils.PoseProviderConfig,
            "from_hypes",
            return_value=cfg_obj,
        ) as from_hypes_mock, mock.patch.object(
            train_utils,
            "apply_pose_provider",
            side_effect=lambda in_batch, cfg: {"cfg_id": id(cfg), "batch": in_batch},
        ) as apply_mock:
            out1 = train_utils.maybe_apply_pose_provider(batch, hypes)
            out2 = train_utils.maybe_apply_pose_provider(batch, hypes)

        self.assertEqual(from_hypes_mock.call_count, 1)
        self.assertEqual(apply_mock.call_count, 2)
        self.assertEqual(out1["cfg_id"], out2["cfg_id"])

    def test_maybe_apply_pose_provider_rebuilds_cache_when_signature_changes(self):
        hypes = {
            "pose_provider": {
                "enabled": True,
                "runtime_mode": "fusion_only",
                "pose_source": "identity",
            }
        }
        batch = {"ego": {}}
        cfg_a = _DummyCfg(enabled=True)
        cfg_b = _DummyCfg(enabled=True)

        with mock.patch.object(
            train_utils.PoseProviderConfig,
            "from_hypes",
            side_effect=[cfg_a, cfg_b],
        ) as from_hypes_mock, mock.patch.object(
            train_utils,
            "apply_pose_provider",
            side_effect=lambda in_batch, cfg: {"cfg_id": id(cfg), "batch": in_batch},
        ):
            out1 = train_utils.maybe_apply_pose_provider(batch, hypes)
            hypes["pose_provider"]["pose_source"] = "gt"
            out2 = train_utils.maybe_apply_pose_provider(batch, hypes)

        self.assertEqual(from_hypes_mock.call_count, 2)
        self.assertNotEqual(out1["cfg_id"], out2["cfg_id"])

    def test_disabled_provider_short_circuits_apply(self):
        hypes = {"pose_provider": {"enabled": False}}
        batch = {"ego": {"x": 1}}

        with mock.patch.object(
            train_utils.PoseProviderConfig,
            "from_hypes",
            return_value=_DummyCfg(enabled=False),
        ) as from_hypes_mock, mock.patch.object(
            train_utils,
            "apply_pose_provider",
            side_effect=RuntimeError("should not be called"),
        ):
            out = train_utils.maybe_apply_pose_provider(batch, hypes)
            out2 = train_utils.maybe_apply_pose_provider(batch, hypes)

        self.assertEqual(from_hypes_mock.call_count, 1)
        self.assertIs(out, batch)
        self.assertIs(out2, batch)


if __name__ == "__main__":
    unittest.main()
