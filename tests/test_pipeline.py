"""The plan and the prepare -> submit -> fetch workflow, with the IonQ API mocked."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import forte
import pipeline
import plan


class PlanTests(unittest.TestCase):
    def test_plan_sizes_and_zne_subset(self):
        circuits, folded = plan.circuits(), plan.zne_circuits()
        self.assertEqual(len(circuits), 51)
        self.assertEqual(len(folded), 13)
        self.assertLessEqual(set(folded), set(circuits))
        self.assertEqual({method for method, *_ in folded}, {"adapted", "pairs", "h_only"})
        self.assertLessEqual(max(p for _, p, _, _ in circuits), 7)

    def test_forgetful_secrets_span_one_orbit_and_another(self):
        self.assertEqual(plan.forgetful_secrets(7, 3), [1, 2, 3])

    def test_job_plan_holds_every_figure_job(self):
        jobs = plan.jobs()
        self.assertEqual(len(jobs), len(set(jobs)))
        self.assertEqual({job[:4] for job in jobs if job[4] == 1}, set(plan.circuits()) | set(plan.NOT_PLOTTED))
        self.assertEqual({job[:4] for job in jobs if job[4] == plan.ZNE_FOLD}, set(plan.zne_circuits()))
        self.assertFalse(set(plan.circuits()) & set(plan.NOT_PLOTTED))
        self.assertEqual(len(jobs), 62+13)  # The paper's 75 hardware jobs.
        # Each circuit's copies are adjacent, so they are submitted one after another.
        self.assertEqual(jobs[:2], [("adapted", 3, 2, 1, 1), ("adapted", 3, 2, 1, plan.ZNE_FOLD)])

    def test_published_hardware_run_is_the_plan(self):
        run = Path(__file__).resolve().parent.parent/"runs"/"hardware"
        names = {f"{method}_p{p}_q{q}_b{b}" + (f"_fold{fold}" if fold != 1 else "")
                 for method, p, q, b, fold in plan.jobs()}
        manifest = json.loads((run/"jobs"/"manifest.json").read_text())
        self.assertEqual({Path(row["file"]).stem for row in manifest} - {"bit_order_probe"}, names)
        for name in names:
            self.assertTrue((run/"results"/f"{name}.npz").exists(), name)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        # Two circuits, one also ZZ-folded, in a scratch runs/ folder.
        jobs = [("pairs", 3, 2, 1, 1), ("pairs", 3, 2, 1, 3), ("weak", 3, 2, 0, 1)]
        for patch in (mock.patch("pipeline.RUNS", self.root),
                      mock.patch("simulation.CACHE", self.root/"aer"),
                      mock.patch("plan.jobs", lambda: jobs)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_prepare_submit_fetch(self):
        pipeline.prepare("emulator")
        run = self.root/"emulator"
        manifest = json.loads((run/"jobs"/"manifest.json").read_text())
        self.assertEqual([row["file"] for row in manifest],
                         ["bit_order_probe.json", "pairs_p3_q2_b1.json", "pairs_p3_q2_b1_fold3.json",
                          "weak_p3_q2_b0.json"])
        for row in manifest:
            payload = json.loads((run/"jobs"/row["file"]).read_text())
            self.assertEqual(payload["noise"]["seed"], forte.NOISE_SEED)
            self.assertFalse(payload["settings"]["error_mitigation"]["debiasing"])  # The emulator cannot.
        with self.assertRaises(FileExistsError):
            pipeline.prepare("emulator")

        sent = []
        def post(path, payload):
            sent.append(payload["name"])
            return dict(id=f"id-{payload['name']}", status="submitted")
        with mock.patch("forte.api_request", side_effect=post):
            pipeline.submit("emulator")
            pipeline.submit("emulator")  # Rerun: every job already has a receipt.
        self.assertEqual(sent, ["bit_order_probe", "pairs_p3_q2_b1", "pairs_p3_q2_b1_fold3", "weak_p3_q2_b0"])

        def histogram(job_id):
            name = job_id.removeprefix("id-")
            payload = json.loads((run/"jobs"/f"{name}.json").read_text())
            width = payload["input"]["qubits"]
            counts = {"1"+"0"*(width-1): payload["shots"]} if name == "bit_order_probe" else \
                     {"0"*width: payload["shots"]-5, "1"*width: 5}
            artifact = dict(histogram=dict(registers=dict(output_all=counts)))
            return dict(status="completed"), artifact
        with mock.patch("forte.fetch_histogram", side_effect=histogram):
            self.assertTrue(pipeline.fetch("emulator"))
        self.assertEqual(json.loads((run/"results"/"bit_order.json").read_text())["bit_order"], "q0-left")
        with np.load(run/"results"/"pairs_p3_q2_b1_fold3.npz") as saved:
            self.assertEqual(int(saved["zne_fold"]), 3)
            self.assertEqual(int(saved["counts"].sum()), plan.SHOTS)
            self.assertEqual(int(saved["k_zero"]), plan.SHOTS-5)  # All-zero label: k = 0.
            self.assertEqual(str(saved["source"]), forte.SOURCES["emulator"])

    def test_hardware_jobs_are_debiased(self):
        pipeline.prepare("hardware")
        run = self.root/"hardware"
        manifest = json.loads((run/"jobs"/"manifest.json").read_text())
        for row in manifest:
            payload = json.loads((run/"jobs"/row["file"]).read_text())
            # The 100-shot bit-order probe is below IonQ's debiasing minimum.
            self.assertEqual(payload["settings"]["error_mitigation"]["debiasing"],
                             row["file"] != "bit_order_probe.json", row["file"])
        # Fetch with receipts written by hand: submission itself stays locked.
        (run/"results").mkdir()
        for row in manifest:
            pipeline.write_json(pipeline.receipt_path(run, row),
                                dict(response=dict(id=f"id-{Path(row['file']).stem}"), file=row["file"]))
        def histogram(job_id):
            name = job_id.removeprefix("id-")
            width = json.loads((run/"jobs"/f"{name}.json").read_text())["input"]["qubits"]
            label = "1"+"0"*(width-1) if name == "bit_order_probe" else "0"*width
            return dict(status="completed"), dict(histogram=dict(registers=dict(output_all={label: 100 if
                        name == "bit_order_probe" else plan.SHOTS})))
        with mock.patch("forte.fetch_histogram", side_effect=histogram):
            self.assertTrue(pipeline.fetch("hardware"))
        with np.load(run/"results"/"weak_p3_q2_b0.npz") as saved:
            self.assertEqual(str(saved["source"]), forte.SOURCES["hardware"])

    def test_changed_job_files_are_refused(self):
        pipeline.prepare("emulator")
        path = self.root/"emulator"/"jobs"/"weak_p3_q2_b0.json"
        path.write_text(path.read_text().replace('"shots": 2048', '"shots": 1'))
        with mock.patch("forte.api_request", return_value=dict(id="x")):
            with self.assertRaisesRegex(ValueError, "changed"):
                pipeline.submit("emulator")

    def test_run_prepared_from_other_circuits_is_refused(self):
        pipeline.prepare("emulator")
        with mock.patch("pipeline.fingerprint", return_value="changed"):
            with mock.patch("forte.api_request") as api:
                with self.assertRaisesRegex(RuntimeError, "different circuits"):
                    pipeline.submit("emulator")
                api.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "different circuits"):
                pipeline.figures("emulator")

    def test_hardware_submission_is_blocked(self):
        pipeline.prepare("hardware")
        with mock.patch("forte.urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "locked"):
                pipeline.submit("hardware")
            with self.assertRaisesRegex(RuntimeError, "locked"):
                pipeline.estimate("hardware")
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
