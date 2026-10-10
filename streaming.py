"""Bounded NDJSON bridge with disconnect cancellation for blocking model code."""
import asyncio
import json
import queue
import threading

from fastapi.responses import StreamingResponse


def event_response(worker, cancel=None):
    cancel = cancel or threading.Event()
    events = queue.Queue(maxsize=32)
    def emit(event):
        while not cancel.is_set():
            try: events.put(event, timeout=.2); return
            except queue.Full: pass
    def run():
        try: worker(emit, cancel)
        finally: emit(None)
    async def stream():
        thread = threading.Thread(target=run, daemon=True); thread.start()
        try:
            while not cancel.is_set():
                try: event = await asyncio.to_thread(events.get, True, .25)
                except queue.Empty: continue
                if event is None: break
                yield json.dumps(event, ensure_ascii=False) + '\n'
        finally: cancel.set()
    return StreamingResponse(stream(), media_type='application/x-ndjson',
        headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
