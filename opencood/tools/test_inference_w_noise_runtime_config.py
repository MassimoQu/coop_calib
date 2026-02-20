import copy
import sys
import unittest
from unittest import mock

from opencood.tools import inference_w_noise as infer_mod


class _DummyModel(object):
    def cuda(self):
        return self

    def eval(self):
        return self


class _DummyDataset(object):
    def __init__(self):
        self.params = {}
        self.pose_override_map = None
        self.pose_override_enabled = False

    def collate_batch_test(self, batch):
        return batch

    def set_pose_override_map(self, overrides):
        self.pose_override_map = overrides
        self.pose_override_enabled = True


class _EmptyLoader(object):
    def __init__(self, *args, **kwargs):
        self._iter = []

    def __iter__(self):
        return iter(self._iter)


class _SolverResult(object):
    def __init__(self):
        self.metrics = {"ok": 1}
        self.overrides = {"0": {"cav_id_list": [0], "lidar_pose_pred_np": [[0, 0, 0, 0, 0, 0]]}}


class InferenceWNoiseRuntimeConfigTest(unittest.TestCase):
    def _base_hypes(self):
        return {
            "test_dir": "OPV2V/test",
            "postprocess": {"gt_range": [-100.0, -40.0, -3.0, 100.0, 40.0, 1.0]},
            "fusion": {"args": {"proj_first": False}},
            "train_params": {"max_cav": 2},
        }

    def _run_main(self, extra_args):
        captured = {}
        run_pose_solver_mock = mock.Mock(return_value=_SolverResult())
        build_pose_corrector_mock = mock.Mock(return_value=object())

        def _fake_load_yaml(_, __):
            return self._base_hypes()

        def _fake_build_dataset(hypes, visualize=True, train=False):
            captured["hypes"] = copy.deepcopy(hypes)
            dataset = _DummyDataset()
            captured["dataset"] = dataset
            return dataset

        argv = [
            "prog",
            "--model_dir",
            "/tmp/fake_model",
            "--fusion_method",
            "intermediate",
            "--pos-std-list",
            "0",
            "--rot-std-list",
            "0",
            "--sweep-mode",
            "paired",
            "--num-workers",
            "0",
            "--max-eval-samples",
            "0",
            "--save_vis_interval",
            "999999",
        ]
        argv.extend(extra_args)

        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            infer_mod.yaml_utils, "load_yaml", side_effect=_fake_load_yaml
        ), mock.patch.object(
            infer_mod.train_utils, "create_model", return_value=_DummyModel()
        ), mock.patch.object(
            infer_mod.train_utils,
            "load_saved_model",
            side_effect=lambda saved_path, model: (0, model),
        ), mock.patch.object(
            infer_mod, "build_dataset", side_effect=_fake_build_dataset
        ), mock.patch.object(
            infer_mod, "DataLoader", _EmptyLoader
        ), mock.patch.object(
            infer_mod.eval_utils,
            "eval_final_results",
            return_value=(0.0, 0.0, 0.0),
        ), mock.patch.object(
            infer_mod.yaml_utils, "save_yaml", side_effect=lambda *a, **k: None
        ), mock.patch.object(
            infer_mod, "resolve_repo_path", side_effect=lambda p: p
        ), mock.patch.object(
            infer_mod, "read_json", return_value={"dummy": True}
        ), mock.patch.object(
            infer_mod, "run_pose_solver", run_pose_solver_mock
        ), mock.patch.object(
            infer_mod, "build_pose_corrector", build_pose_corrector_mock
        ), mock.patch.object(
            infer_mod.torch.cuda, "is_available", return_value=False
        ):
            infer_mod.main()

        captured["run_pose_solver_calls"] = run_pose_solver_mock.call_count
        captured["build_pose_corrector_calls"] = build_pose_corrector_mock.call_count
        return captured

    def test_online_backend_injects_pose_provider(self):
        out = self._run_main(
            [
                "--pose-correction",
                "v2xregpp_initfree",
                "--stage1-result",
                "/tmp/stage1.json",
                "--solver-backend",
                "online_box",
                "--runtime-mode",
                "register_and_fuse",
                "--pose-source",
                "noisy_input",
            ]
        )
        hypes = out["hypes"]
        pose_provider = hypes.get("pose_provider") or {}
        self.assertTrue(bool(pose_provider.get("enabled")))
        self.assertEqual(str(pose_provider.get("runtime_mode")), "register_and_fuse")
        self.assertEqual(str(pose_provider.get("solver_backend")), "online_box")
        self.assertEqual(str(pose_provider.get("online_method")), "v2xregpp")
        self.assertEqual(str(pose_provider.get("stage1_result")), "/tmp/stage1.json")
        self.assertEqual(str((pose_provider.get("online_args") or {}).get("mode")), "initfree")
        self.assertEqual(bool(hypes.get("comm_range_use_clean_pose", False)), True)
        self.assertEqual(bool((hypes.get("pose_override") or {}).get("enabled", True)), False)
        self.assertEqual(int(out["run_pose_solver_calls"]), 0)

    def test_offline_backend_runs_solver_and_uses_override_map(self):
        out = self._run_main(
            [
                "--pose-correction",
                "v2xregpp_initfree",
                "--stage1-result",
                "/tmp/stage1.json",
                "--solver-backend",
                "offline_map",
            ]
        )
        hypes = out["hypes"]
        pose_override = hypes.get("pose_override") or {}
        self.assertTrue(bool(pose_override.get("enabled", False)))
        self.assertEqual(int(out["build_pose_corrector_calls"]), 1)
        self.assertEqual(int(out["run_pose_solver_calls"]), 1)
        dataset = out["dataset"]
        self.assertTrue(bool(dataset.pose_override_enabled))
        self.assertFalse(bool((dataset.params.get("noise_setting") or {}).get("add_noise", True)))


    def test_online_oracle_gt_uses_no_noise_dataset_path(self):
        out = self._run_main(
            [
                "--pose-correction",
                "oracle_gt",
                "--solver-backend",
                "online_box",
                "--runtime-mode",
                "register_and_fuse",
            ]
        )
        dataset = out["dataset"]
        noise = dataset.params.get("noise_setting") or {}
        self.assertEqual(bool(noise.get("add_noise", True)), False)
        pose_provider = (out["hypes"].get("pose_provider") or {})
        self.assertEqual(str(pose_provider.get("online_method")), "gt")

    def test_runtime_mode_without_pose_correction_sets_pose_provider(self):
        out = self._run_main(
            [
                "--runtime-mode",
                "fusion_only",
                "--solver-backend",
                "offline_map",
                "--pose-source",
                "identity",
            ]
        )
        hypes = out["hypes"]
        pose_provider = hypes.get("pose_provider") or {}
        self.assertTrue(bool(pose_provider.get("enabled")))
        self.assertEqual(str(pose_provider.get("runtime_mode")), "fusion_only")
        self.assertEqual(str(pose_provider.get("solver_backend")), "offline_map")
        self.assertEqual(str(pose_provider.get("pose_source")), "identity")
        self.assertEqual(int(out["run_pose_solver_calls"]), 0)


if __name__ == "__main__":
    unittest.main()
