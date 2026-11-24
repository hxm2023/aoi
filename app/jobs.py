"""Single-worker background job queue for heavy region runs."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid


class JobManager:
    def __init__(self, max_finished=8):
        self._q = queue.Queue()
        self._jobs = {}
        self._lock = threading.Lock()
        self._thread = None
        self._max_finished = max_finished

    def _start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, fn, meta=None):
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._jobs[jid] = {
                "jid": jid, "status": "queued", "stage": "排队中",
                "pct": 0, "message": "", "created": now,
                "updated": now, "error": None, "result": None,
                "meta": meta or {}, "_fn": fn, "elapsed_s": 0.0,
            }
        self._start()
        self._q.put(jid)
        return jid

    def _loop(self):
        while True:
            jid = self._q.get()
            with self._lock:
                job = self._jobs.get(jid)
                if job is None:
                    continue
                job.update(status="running", stage="启动", pct=1,
                           updated=time.time())
                fn = job["_fn"]
                t0 = job["created"]

            def progress(stage, pct, message=None):
                with self._lock:
                    j = self._jobs.get(jid)
                    if j is None:
                        return
                    j.update(stage=str(stage), pct=int(pct),
                             message="" if message is None else str(message),
                             updated=time.time())

            try:
                result = fn(progress)
                with self._lock:
                    j = self._jobs[jid]
                    j.update(status="done", stage="完成", pct=100,
                             message="", result=result,
                             updated=time.time(),
                             elapsed_s=round(time.time() - t0, 1))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                with self._lock:
                    j = self._jobs[jid]
                    j.update(status="error", stage="失败", error=str(exc),
                             updated=time.time(),
                             elapsed_s=round(time.time() - t0, 1))
            finally:
                with self._lock:
                    if jid in self._jobs:
                        self._jobs[jid]["_fn"] = None
                    self._cleanup()

    def _cleanup(self):
        done = [j for j in self._jobs.values()
                if j["status"] in ("done", "error")]
        done.sort(key=lambda j: j["updated"])
        for j in done[:-self._max_finished]:
            self._jobs.pop(j["jid"], None)

    def get(self, jid):
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return None
            out = {k: v for k, v in job.items() if not k.startswith("_")}
            return out
