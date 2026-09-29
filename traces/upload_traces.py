"""Load downloaded traces, spread timestamps over a recent window, regenerate IDs, and upload.

Run from the traces folder with:

    uv run python3 upload_traces.py
    uv run python3 upload_traces.py --project my-project --input traces.json
    uv run python3 upload_traces.py --days 0.5 --seed 42
"""

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, uuid7
from langsmith.utils import LangSmithNotFoundError

# Project name comes from .env (LANGSMITH_PROJECT); override with --project.
DEFAULT_PROJECT = os.getenv("LANGSMITH_PROJECT")

# Ingest rejects any run whose start_time is more than 24 hours from now, on both
# the multipart and batch endpoints, with a 422. That caps how far back traces can
# be dated: ask for more and the runs are silently dropped. Stay under the limit
# so the last trace in the spread still clears it once the upload has run a while.
MAX_BACKDATE_DAYS = 0.95

# Ingested runs take a moment to become queryable, so the landing check retries
# rather than failing on the first empty read.
WAIT_ATTEMPTS = 10
WAIT_SECONDS = 3
REUPLOAD_ATTEMPTS = 2
RUN_ID_BATCH_SIZE = 100
DEFAULT_MANIFEST = "upload_manifest.json"

# The thumb a rep leaves on a reply. Reps rate what is in front of them: they
# asked for something, and either it happened or it visibly did not. A send that
# goes through reads as a win at the moment of rating, so it gets the thumb.
# Anything the rep learns later -- from the account team, from a reply that never
# comes -- lands well after the rating is written, and nobody goes back to revise
# it. Most ratings carry no note, which is how reps actually use the thumbs.
RATING_KEY = "rep_rating"
GOOD_NOTES = (None, None, None, None, None, None,
              "looks good, sent", "perfect, thanks", "good to go", "yep this works")
BAD_NOTES = (None, None, None, None, None,
             "this never actually went out", "errored on me, had to do it by hand")


def parse_dt(s):
    """Parse an ISO timestamp string into a naive (tz-stripped) datetime."""
    if s is None:
        return None
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def tool_message(run):
    """The tool message a tool run returned, or None if it did not return one."""
    output = (run.get("outputs") or {}).get("output")
    return output if isinstance(output, dict) else None


def tool_payloads(trace_runs, tool_name):
    """Yield the decoded payload of every completed call to the named tool."""
    for run in trace_runs:
        if run["run_type"] != "tool" or run["name"] != tool_name:
            continue
        message = tool_message(run)
        try:
            payload = json.loads(message["content"])
        except (KeyError, TypeError, ValueError):
            # A tool that failed mid-call returns prose, not a payload.
            continue
        if isinstance(payload, dict):
            yield payload


def visibly_failed(trace_runs):
    """Did anything go wrong that the rep would notice while reading the reply?"""
    for run in trace_runs:
        # An error on any run surfaces in the reply the rep reads, so check every
        # run rather than only the tools: a chain or model failure is just as
        # visible, and only the tools carry a status field to check as well.
        if run.get("error"):
            return True
        if run["run_type"] != "tool":
            continue
        message = tool_message(run)
        if message and message.get("status") == "error":
            return True
    return any(
        payload.get("status") != "sent"
        for payload in tool_payloads(trace_runs, "send_prospect_email")
    )


def rate_trace(trace_runs):
    """Return the (score, comment) a rep would leave on this trace. 1 = good, 0 = bad."""
    if visibly_failed(trace_runs):
        return 0, random.choice(BAD_NOTES)
    return 1, random.choice(GOOD_NOTES)


def bootstrap_project(client, name):
    """Materialize the project up front so downstream reads don't race propagation.

    On a fresh project, read_project 404s until the first run has landed and been
    indexed -- so the verification step at the end of the upload would crash even
    though the ingest itself succeeded. Creating (or confirming) the project here
    means everything that follows can trust that read_project resolves, and a real
    auth/tenant problem surfaces before we upload hundreds of runs.
    """
    try:
        return client.read_project(project_name=name).id
    except LangSmithNotFoundError:
        return client.create_project(project_name=name).id


def write_manifest(path, traces):
    """Write expected regenerated run IDs and types for each trace."""
    manifest = {
        "traces": {
            trace_id: {
                "runs": [
                    {"id": run["id"], "run_type": run["run_type"]}
                    for run in trace_runs
                ]
            }
            for trace_id, trace_runs in traces.items()
        }
    }
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    """Load an upload manifest from disk."""
    with open(path) as f:
        manifest = json.load(f)
    if not isinstance(manifest.get("traces"), dict):
        raise ValueError(f"Manifest {path} must contain a traces object.")
    return manifest


def list_landed_runs(client, project_id, expected_ids):
    """Return landed run IDs and types using bounded ID-filtered queries."""
    landed = {}
    expected_ids = list(expected_ids)
    for start in range(0, len(expected_ids), RUN_ID_BATCH_SIZE):
        for run in client.list_runs(
            project_id=project_id,
            run_ids=expected_ids[start : start + RUN_ID_BATCH_SIZE],
            select=["id", "name", "start_time", "run_type", "trace_id"],
        ):
            landed[str(run.id)] = run.run_type
    return landed


def incomplete_traces(manifest, landed):
    """Return per-trace ID and run-type differences from a landing check."""
    incomplete = []
    for trace_id, trace in manifest["traces"].items():
        expected_runs = trace["runs"]
        expected = {run["id"]: run["run_type"] for run in expected_runs}
        landed_ids = set(expected) & set(landed)
        expected_types = Counter(expected.values())
        landed_types = Counter(landed[run_id] for run_id in landed_ids)
        missing_types = {
            run_type: count - landed_types[run_type]
            for run_type, count in expected_types.items()
            if count > landed_types[run_type]
        }
        unexpected_types = {
            run_type: count - expected_types[run_type]
            for run_type, count in landed_types.items()
            if count > expected_types[run_type]
        }
        if len(landed_ids) != len(expected) or missing_types or unexpected_types:
            incomplete.append(
                {
                    "trace_id": trace_id,
                    "expected_ids": set(expected),
                    "landed_ids": landed_ids,
                    "missing_ids": set(expected) - landed_ids,
                    "expected_count": len(expected),
                    "landed_count": len(landed_ids),
                    "missing_types": missing_types,
                    "unexpected_types": unexpected_types,
                }
            )
    return incomplete


def verify_manifest(client, project_id, manifest):
    """Wait for and report incomplete traces in an existing project."""
    expected_ids = {
        run["id"]
        for trace in manifest["traces"].values()
        for run in trace["runs"]
    }
    incomplete = []
    for attempt in range(WAIT_ATTEMPTS):
        landed = list_landed_runs(client, project_id, expected_ids)
        incomplete = incomplete_traces(manifest, landed)
        if not incomplete:
            return []
        if attempt + 1 < WAIT_ATTEMPTS:
            time.sleep(WAIT_SECONDS)
    return incomplete


def format_incomplete_traces(incomplete):
    """Format all incomplete trace details for an upload failure."""
    return "; ".join(
        f"{item['trace_id']}: {item['landed_count']}/{item['expected_count']} runs, "
        f"missing types {item['missing_types'] or {}}"
        for item in incomplete
    )


def create_run(client, run, project_name):
    """Enqueue one run with the regenerated trace data."""
    client.create_run(
        id=run["id"],
        trace_id=run["trace_id"],
        dotted_order=run["dotted_order"],
        parent_run_id=run["parent_run_id"],
        name=run["name"],
        run_type=run["run_type"],
        inputs=run["inputs"],
        outputs=run.get("outputs"),
        error=run.get("error"),
        extra=run.get("extra"),
        tags=run.get("tags"),
        start_time=run["start_time"],
        end_time=run["end_time"],
        project_name=project_name,
    )


def report_incomplete_traces(incomplete):
    """Raise an upload error containing every incomplete trace."""
    raise SystemExit(
        "ERROR: incomplete traces after upload: "
        + format_incomplete_traces(incomplete)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Target project name")
    parser.add_argument("--input", default="traces.json", help="Input file path")
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--verify-manifest",
        help="Audit an existing project from a manifest without uploading",
    )
    parser.add_argument(
        "--days",
        type=float,
        default=MAX_BACKDATE_DAYS,
        help=f"Spread traces randomly over this many days ending now "
        f"(default: {MAX_BACKDATE_DAYS}; the ingest API rejects anything older than 24h)",
    )
    parser.add_argument("--seed", type=int, help="Seed for a reproducible upload")
    args = parser.parse_args()

    if args.days > MAX_BACKDATE_DAYS:
        parser.error(
            f"--days {args.days} exceeds the ingest API's 24-hour backdating limit; "
            f"runs older than that are rejected with a 422 and never land. "
            f"Use --days {MAX_BACKDATE_DAYS} or less."
        )

    if args.seed is not None:
        random.seed(args.seed)

    if not args.project:
        parser.error("No project name found. Set LANGSMITH_PROJECT in .env or pass --project.")

    if args.verify_manifest:
        client = Client()
        project_id = client.read_project(project_name=args.project).id
        manifest = load_manifest(args.verify_manifest)
        incomplete = verify_manifest(client, project_id, manifest)
        if incomplete:
            report_incomplete_traces(incomplete)
        print(f"Verified {len(manifest['traces'])} traces from {args.verify_manifest}.")
        return

    with open(args.input) as f:
        runs = json.load(f)

    print(f"Loaded {len(runs)} runs from {args.input}")
    if not runs:
        print("Nothing to upload.")
        return

    start_times = [parse_dt(r["start_time"]) for r in runs if r.get("start_time")]
    if not start_times:
        raise ValueError("No runs have a start_time; cannot compute time shift.")

    # Build a map from old IDs to fresh uuid7s (uuid7 is time-ordered).
    # For root runs, trace_id must equal id, so map both to the same new uuid7.
    id_map = {}
    for run in runs:
        if run.get("parent_run_id") is None:
            root_new_id = str(uuid7())
            id_map[run["id"]] = root_new_id
            id_map[run["trace_id"]] = root_new_id
    for run in runs:
        for field in ("id", "parent_run_id"):
            old_id = run.get(field)
            if old_id and old_id not in id_map:
                id_map[old_id] = str(uuid7())

    # Group runs by (new) trace id, keeping original times for now.
    traces = defaultdict(list)
    for run in runs:
        trace_id = id_map[run["trace_id"]]
        traces[trace_id].append(
            {
                "id": id_map[run["id"]],
                "trace_id": trace_id,
                "dotted_order": None,  # populated below
                "parent_run_id": id_map.get(run.get("parent_run_id")),
                "name": run["name"],
                "run_type": run["run_type"],
                "inputs": run.get("inputs") or {},
                "outputs": run.get("outputs"),
                "error": run.get("error"),
                "extra": run.get("extra") or {},
                "tags": run.get("tags"),
                "start_time": parse_dt(run["start_time"]),
                "end_time": parse_dt(run["end_time"]) if run.get("end_time") else None,
                # Carried through the id remap and replayed once the runs exist.
                # Absent from traces captured before feedback was collected.
                "feedback": run.get("feedback") or [],
            }
        )

    # Scatter each trace to a random point in the last `--days` days, shifting all of
    # its runs by the same delta so the internal spacing/nesting of a trace is intact.
    # Traces land independently of one another, so the batch looks like organic
    # day-to-day usage rather than one replayed burst.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    window = timedelta(days=args.days)
    for trace_runs in traces.values():
        trace_start = min(r["start_time"] for r in trace_runs)
        trace_end = max(
            [r["end_time"] for r in trace_runs if r["end_time"]] or [trace_start]
        )
        duration = trace_end - trace_start
        # Keep the whole trace inside the window (nothing ends in the future).
        latest_start = max(window - duration, timedelta(0))
        offset = timedelta(seconds=random.uniform(0, latest_start.total_seconds()))
        delta = (now - window + offset) - trace_start
        for run in trace_runs:
            run["start_time"] += delta
            if run["end_time"]:
                run["end_time"] += delta

    all_starts = [r["start_time"] for trace_runs in traces.values() for r in trace_runs]
    print(
        f"Spread {len(traces)} traces over {args.days} days: "
        f"{min(all_starts):%Y-%m-%d %H:%M} to {max(all_starts):%Y-%m-%d %H:%M}"
    )
    write_manifest(args.manifest, traces)

    client = Client()
    project_id = bootstrap_project(client, args.project)
    print(f"Uploading {len(traces)} traces to project '{args.project}'...")

    for i, (trace_id, trace_runs) in enumerate(traces.items()):
        # Sort: root first, then children by start_time.
        trace_runs.sort(key=lambda r: (r["parent_run_id"] is not None, r["start_time"]))

        # Build dotted_order by walking the parent chain, so nesting is correct
        # regardless of run order or start_time skew.
        runs_by_id = {run["id"]: run for run in trace_runs}
        dotted_orders = {}

        def build_dotted_order(run):
            rid = run["id"]
            if rid in dotted_orders:
                return dotted_orders[rid]
            ts = run["start_time"].strftime("%Y%m%dT%H%M%S%f") + "Z"
            segment = f"{ts}{rid}"
            parent = runs_by_id.get(run["parent_run_id"])
            order = segment if parent is None else f"{build_dotted_order(parent)}.{segment}"
            dotted_orders[rid] = order
            run["dotted_order"] = order
            return order

        for run in trace_runs:
            build_dotted_order(run)

        for run in trace_runs:
            create_run(client, run, args.project)

        if (i + 1) % 10 == 0:
            print(f"  Uploaded {i + 1}/{len(traces)} traces")

    # Wait for all background operations to complete.
    print("Flushing...")
    client.flush()

    manifest = load_manifest(args.manifest)
    incomplete = verify_manifest(client, project_id, manifest)
    run_by_id = {
        run["id"]: run for trace_runs in traces.values() for run in trace_runs
    }
    for _ in range(REUPLOAD_ATTEMPTS):
        if not incomplete:
            break
        missing_ids = {
            run_id for item in incomplete for run_id in item["missing_ids"]
        }
        if not missing_ids:
            break
        for run_id in missing_ids:
            create_run(client, run_by_id[run_id], args.project)
        client.flush()
        incomplete = verify_manifest(client, project_id, manifest)
    if incomplete:
        report_incomplete_traces(incomplete)
    expected_count = sum(len(trace["runs"]) for trace in manifest["traces"].values())
    print(f"Verified {expected_count}/{expected_count} runs landed.")

    # Every trace carries the rating its rep left on the reply. Traces captured
    # before ratings were collected have none, so derive the one that rep would
    # have left; traces that already carry theirs keep it rather than being rated
    # a second time.
    for trace_runs in traces.values():
        root = next(r for r in trace_runs if r["parent_run_id"] is None)
        if any(f["key"] == RATING_KEY for f in root["feedback"]):
            continue
        score, comment = rate_trace(trace_runs)
        root["feedback"].append(
            {"key": RATING_KEY, "score": score, "value": None, "comment": comment}
        )

    # Feedback is keyed by run id, so it can only be attached once the runs it
    # points at have been created -- hence after the verification above, and against
    # the regenerated ids rather than the ones in the input file.
    #
    # Passing trace_id puts each record on the batched tracing queue instead of a
    # blocking POST per record; the flush below waits for them.
    n_feedback = 0
    for trace_runs in traces.values():
        for run in trace_runs:
            for feedback in run["feedback"]:
                client.create_feedback(
                    run_id=run["id"],
                    trace_id=run["trace_id"],
                    key=feedback["key"],
                    score=feedback.get("score"),
                    value=feedback.get("value"),
                    comment=feedback.get("comment"),
                )
                n_feedback += 1
    if n_feedback:
        client.flush()

    print(f"Done! Uploaded {len(traces)} traces to '{args.project}'.")


if __name__ == "__main__":
    main()
