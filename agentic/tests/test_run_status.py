"""检查续跑进度、最终零待办值和失败回答的队列清理。"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch


class RunStatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # server 导入时会初始化 SQLite；全部测试数据都放入临时目录。
        cls.directory = tempfile.mkdtemp(prefix="urbanvideo-status-")
        previous = os.getcwd()
        try:
            os.chdir(cls.directory)
            import server
        finally:
            os.chdir(previous)
        cls.server = server

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def setUp(self):
        self.result_directory = tempfile.mkdtemp(prefix="urbanvideo-result-")
        self.result_patch = patch.object(self.server, "RESULT_DIR", self.result_directory)
        self.result_patch.start()
        self.addCleanup(self.result_patch.stop)
        self.addCleanup(lambda: shutil.rmtree(self.result_directory, ignore_errors=True))
        self.server.run_status_by_run.clear()
        self.server.active_states.clear()
        self.client = self.server.app.test_client()
        self.run_id = "resume-progress-test"
        self.server.init_db(self.run_id)
        self.client.post("/update", json={
            "run_id": self.run_id, "run_state": "running",
            "total_questions": 3, "processed_questions": 1, "pending_questions": 2,
        })

    def status(self):
        return self.client.get(f"/state?run_id={self.run_id}").get_json()["run_status"]

    def test_completed_with_pending_questions_is_stopped(self):
        self.client.post("/update", json={
            "run_id": self.run_id, "run_state": "completed",
            "total_questions": 3, "processed_questions": 2, "pending_questions": 1,
        })
        self.assertEqual(self.status()["run_state"], "stopped")

    def test_final_zero_is_preserved(self):
        self.client.post("/update", json={
            "run_id": self.run_id, "run_state": "completed",
            "total_questions": 3, "processed_questions": 3, "pending_questions": 0,
        })
        self.assertEqual(self.status()["pending_questions"], 0)
        self.assertEqual(self.status()["run_state"], "completed")

    def test_recomputed_history_uses_explicit_csv_progress(self):
        payload = {
            "run_id": self.run_id, "qa_idx": 1, "model_answer": "A",
            "run_state": "running", "total_questions": 3,
            "processed_questions": 2, "pending_questions": 1,
        }
        with patch.object(self.server, "save_history_jsonl_compat"):
            self.client.post("/update", json=payload)
            # 重试或旧数据库中已有题号时，不能再按插入次数加一。
            self.client.post("/update", json=payload)
        self.assertEqual(self.status()["processed_questions"], 2)
        self.assertEqual(self.status()["pending_questions"], 1)

    def test_none_answer_clears_active_queue_without_creating_history(self):
        self.client.post("/update", json={"run_id": self.run_id, "qa_idx": 2, "stage": "抽帧"})
        self.client.post("/update", json={"run_id": self.run_id, "qa_idx": 2, "model_answer": None})
        self.assertEqual(self.server.active_queue(self.run_id), [])
        self.assertEqual(self.server.db_count(self.run_id), 0)
        self.assertEqual(self.status()["pending_questions"], 2)
