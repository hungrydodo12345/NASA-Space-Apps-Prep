"""Resumable multiprocessing job runner: jobs are dicts with a unique 'id'; results append to a JSONL file."""
import os, sys, time, traceback, warnings
from multiprocessing import Pool
from . import common as C


def _work(args):
    fn, job = args
    warnings.filterwarnings("ignore")
    os.environ["OMP_NUM_THREADS"] = "1"
    t = time.time()
    try:
        res = fn(job)
        return dict(id=job["id"], ok=True, sec=time.time() - t, **res)
    except Exception as e:
        return dict(id=job["id"], ok=False, err=repr(e), tb=traceback.format_exc()[-600:])


def run(fn, jobs, out, procs=4):
    done = {r["id"] for r in C.read(out) if r.get("ok")}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{out}: {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time(); n = 0
    with Pool(procs) as p:
        for r in p.imap_unordered(_work, [(fn, j) for j in todo], chunksize=1):
            C.log(out, r); n += 1
            if n % 20 == 0 or not r["ok"]:
                print(f"  {n}/{len(todo)} [{time.time()-t0:.0f}s]" + ("" if r["ok"] else f" FAIL {r['err']}"), flush=True)
    print(f"finished {out} in {time.time()-t0:.0f}s", flush=True)
