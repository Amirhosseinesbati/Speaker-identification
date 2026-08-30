"""Budgeted, Git-pinned experiment supervisor for the Vast worker.

The supervisor intentionally accepts experiment profile names, not arbitrary
shell commands.  It records a durable state transition before and after each
run, refuses dirty/out-of-sync code by default, enforces time/cost guards and
sends concise Persian Telegram milestones without making Telegram a training
dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.campaign_state import CampaignStateError, CampaignStore  # noqa: E402


DEFAULT_STATE = ROOT / "data" / "experiments" / "campaign_state.json"
DEFAULT_EVENTS = ROOT / "data" / "experiments" / "campaign_events.jsonl"
LOG_DIR = ROOT / "data" / "experiments" / "campaign_logs"
MLFLOW_SECRET_FILE = Path("/root/.iaaa_mlflow.env")
MLFLOW_SECRET_KEYS = {
    "DAGSHUB_USER_TOKEN",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TRACKING_URI",
}


def _dotenv_value(raw_value: str) -> str:
    """Decode the common quoted-value subset without evaluating shell syntax."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _notify(message: str) -> None:
    try:
        from telegram_notifier import send

        send(message)
    except Exception as exc:
        # Notification is best-effort and must never kill a training run.
        print(f"Telegram notification warning: {type(exc).__name__}", flush=True)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip()


def _worker_environment() -> dict[str, str]:
    """Return child env plus locked-down MLflow settings, without logging values."""
    environment = dict(os.environ)
    if not MLFLOW_SECRET_FILE.exists():
        return environment
    mode = stat.S_IMODE(MLFLOW_SECRET_FILE.stat().st_mode)
    if mode & 0o077:
        raise CampaignStateError(
            f"MLflow secret file permissions are too broad: {mode:o}"
        )
    for line in MLFLOW_SECRET_FILE.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        decoded = _dotenv_value(value) if separator else ""
        if separator and key in MLFLOW_SECRET_KEYS and decoded:
            environment[key] = decoded
    missing = sorted(MLFLOW_SECRET_KEYS - environment.keys())
    if missing:
        raise CampaignStateError(
            "MLflow tracking secret is incomplete: " + ", ".join(missing)
        )
    environment.setdefault(
        "MLFLOW_TRACKING_URI", environment["DAGSHUB_TRACKING_URI"],
    )
    return environment


def _assert_reproducible_checkout(allow_dirty: bool) -> str:
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    if dirty and not allow_dirty:
        raise CampaignStateError(
            "tracked worker files are dirty; commit and push before a scientific run"
        )
    try:
        divergence = _git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
        behind, ahead = (int(value) for value in divergence.split())
    except (subprocess.CalledProcessError, ValueError):
        raise CampaignStateError("current branch has no valid upstream tracking branch")
    if (behind or ahead) and not allow_dirty:
        raise CampaignStateError(
            f"worker is not synchronized with upstream: behind={behind}, ahead={ahead}"
        )
    return commit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


RECEIPT_SUFFIXES = {".pt", ".npz", ".json", ".yaml", ".yml", ".md"}


def _artifact_receipts(
    profile: str, extra_paths: tuple[Path, ...] = ()
) -> list[dict[str, Any]]:
    base = ROOT / "checkpoints" / profile
    candidates = []
    if base.exists():
        candidates.extend(
            path for path in base.rglob("*")
            if path.is_file() and path.suffix.lower() in RECEIPT_SUFFIXES
        )
    candidates.extend(path for path in extra_paths if path.is_file())
    receipts = []
    seen: set[Path] = set()
    for path in sorted(candidates, key=lambda item: str(item)):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        receipts.append({
            "path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return receipts


def _store(args: argparse.Namespace) -> CampaignStore:
    return CampaignStore(Path(args.state), Path(args.events))


def _status_payload(store: CampaignStore) -> dict[str, Any]:
    state = store.load()
    return {**state, "budget": store.budget_snapshot(state)}


def cmd_init(args: argparse.Namespace) -> int:
    commit = _assert_reproducible_checkout(args.allow_dirty)
    state = _store(args).initialize(
        campaign_id=args.campaign_id,
        instance_id=args.instance_id,
        hourly_rate_usd=args.hourly_rate,
        max_campaign_cost_usd=args.max_cost,
        max_run_hours=args.max_run_hours,
        waiting_keepalive_hours=args.waiting_keepalive_hours,
        billing_started_at=args.billing_started_at,
        git_commit=commit,
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))
    _notify(
        "🧭 ناظر کمپین فعال شد\n\n"
        f"شناسهٔ کمپین: {args.campaign_id}\n"
        f"شناسهٔ نمونهٔ محاسباتی: {args.instance_id}\n"
        f"نسخهٔ کد: {commit[:8]}\n"
        f"سقف هزینه: ${args.max_cost:.2f}\n"
        f"حداکثر زمان هر اجرا: {args.max_run_hours:g} ساعت\n\n"
        "وضعیت: پیش‌پرواز"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    print(json.dumps(_status_payload(_store(args)), indent=2, ensure_ascii=False))
    return 0


def cmd_transition(args: argparse.Namespace) -> int:
    state = _store(args).transition(args.target, args.reason)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def cmd_set_max_run_hours(args: argparse.Namespace) -> int:
    state = _store(args).set_max_run_hours(args.hours, args.reason)
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    store = _store(args)
    state = store.load()
    if state["status"] not in {"READY", "ANALYZING"}:
        raise CampaignStateError(
            f"cannot start an experiment from state {state['status']}"
        )
    commit = _assert_reproducible_checkout(args.allow_dirty)
    reserve_hours = min(args.timeout_hours, state["policy"]["max_run_hours"])
    store.assert_budget(reserve_hours=reserve_hours)

    config_path = ROOT / "configs" / "experiments" / f"{args.profile}.yaml"
    if not config_path.is_file():
        raise CampaignStateError(f"unknown experiment profile: {args.profile}")
    config_sha = _sha256(config_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{stamp}_{args.profile}.log"
    command = [
        sys.executable, "-X", "utf8", "-m", "src.pipelines.run_pipeline",
        "--experiment", args.profile, "--run", args.stage,
    ]
    if args.no_mlflow:
        command.append("--no-mlflow")

    run_record = {
        "profile": args.profile,
        "stage": args.stage,
        "git_commit": commit,
        "config_sha256": config_sha,
        "log_path": str(log_path.relative_to(ROOT)),
        "command": command,
    }
    store.start_run(run_record)
    _notify(
        "▶️ آزمایش جدید آغاز شد\n\n"
        f"پروفایل: {args.profile}\n"
        f"مرحله: {args.stage}\n"
        f"نسخهٔ کد: {commit[:8]}\n"
        f"سقف زمان: {args.timeout_hours:g} ساعت\n\n"
        "پس از پایان، معیارها و artifactهای معتبر گزارش می‌شوند."
    )

    timeout_seconds = 3600 * min(
        args.timeout_hours, float(state["policy"]["max_run_hours"]),
    )
    exit_code = 1
    reason = "experiment failed before process start"
    telegram_reason = "فرایند آموزش پیش از آغاز اجرای اصلی متوقف شد"
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=_worker_environment(),
                timeout=timeout_seconds,
                check=False,
            )
        exit_code = int(process.returncode)
        reason = (
            f"experiment {args.profile} completed"
            if exit_code == 0
            else f"experiment {args.profile} failed with exit code {exit_code}"
        )
        telegram_reason = (
            "اجرا با موفقیت کامل شد"
            if exit_code == 0
            else f"فرایند آموزش با کد خروج {exit_code} متوقف شد"
        )
    except subprocess.TimeoutExpired:
        exit_code = 124
        reason = f"experiment {args.profile} exceeded its time limit"
        telegram_reason = "زمان اجرای آزمایش از سقف مجاز عبور کرد"

    success = exit_code == 0
    # Preserve every recoverable artifact even for timeout/failure. Receipt
    # presence never implies validity; downstream audits still verify hashes,
    # readability, scientific provenance and completeness before promotion.
    artifacts = _artifact_receipts(
        args.profile, extra_paths=(config_path, log_path)
    )
    final_state = store.finish_run(
        success=success,
        exit_code=exit_code,
        reason=reason,
        artifacts=artifacts,
    )
    budget = store.budget_snapshot(final_state)
    if success:
        _notify(
            "✅ آزمایش با موفقیت تمام شد\n\n"
            f"پروفایل: {args.profile}\n"
            f"خروجی‌های ثبت‌شده: {len(artifacts)}\n"
            f"هزینهٔ تخمینی کمپین: ${budget['estimated_cost_usd']:.2f}\n"
            f"گزارش اجرا: {log_path.relative_to(ROOT)}\n\n"
            "وضعیت: تحلیل عمیق نتیجه"
        )
    else:
        _notify(
            "⛔ آزمایش متوقف شد\n\n"
            f"پروفایل: {args.profile}\n"
            f"کد خروج: {exit_code}\n"
            f"علت: {telegram_reason}\n"
            f"گزارش: {log_path.relative_to(ROOT)}\n\n"
            "هیچ اجرای بعدی تا تحلیل علت آغاز نمی‌شود."
        )
    print(json.dumps(_status_payload(store), indent=2, ensure_ascii=False))
    return exit_code


def cmd_wait(args: argparse.Namespace) -> int:
    store = _store(args)
    state = store.load()
    if state["status"] != "ANALYZING":
        raise CampaignStateError("package can be promoted only from ANALYZING")
    artifact = Path(args.artifact).resolve()
    if not artifact.is_file():
        raise CampaignStateError(f"submission artifact not found: {artifact}")
    receipt = {"path": str(artifact), "sha256": _sha256(artifact),
               "size_bytes": artifact.stat().st_size}
    state = store.transition(
        "WAITING_FOR_LEADERBOARD",
        f"submission package promoted: {artifact.name}",
        {"artifact": receipt},
    )
    _notify(
        "📦 بستهٔ پیشنهادی آمادهٔ لیدربرد است\n\n"
        f"نام: {artifact.name}\n"
        f"حجم: {artifact.stat().st_size / 2**20:.1f} مگابایت\n"
        f"SHA256: {receipt['sha256'][:16]}…\n\n"
        "وضعیت: انتظار برای نتیجهٔ واقعی لیدربرد\n"
        "سرور روشن می‌ماند و پس از دریافت امتیاز تحلیل ادامه پیدا می‌کند."
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    store = _store(args)
    state = store.record_leaderboard(args.score, args.note)
    _notify(
        "🏁 نتیجهٔ لیدربرد دریافت شد\n\n"
        f"امتیاز: {args.score:.6f}\n"
        f"یادداشت: {args.note or '—'}\n\n"
        "وضعیت: تحلیل اختلاف پیش‌بینی محلی و امتیاز واقعی و طراحی گام بعدی"
    )
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--events", default=str(DEFAULT_EVENTS))
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--campaign-id", required=True)
    init.add_argument("--instance-id", required=True)
    init.add_argument("--hourly-rate", type=float, required=True)
    init.add_argument("--max-cost", type=float, required=True)
    init.add_argument("--max-run-hours", type=float, default=12.0)
    init.add_argument("--waiting-keepalive-hours", type=float, default=12.0)
    init.add_argument("--billing-started-at", default=None)
    init.add_argument("--allow-dirty", action="store_true")
    init.set_defaults(func=cmd_init)

    status = subparsers.add_parser("status")
    status.set_defaults(func=cmd_status)

    transition = subparsers.add_parser("transition")
    transition.add_argument("target")
    transition.add_argument("--reason", required=True)
    transition.set_defaults(func=cmd_transition)

    max_run_hours = subparsers.add_parser("set-max-run-hours")
    max_run_hours.add_argument("--hours", required=True, type=float)
    max_run_hours.add_argument("--reason", required=True)
    max_run_hours.set_defaults(func=cmd_set_max_run_hours)

    run = subparsers.add_parser("run")
    run.add_argument("--profile", required=True)
    run.add_argument("--stage", choices=["all", "data", "train", "eval"], default="eval")
    run.add_argument("--timeout-hours", type=float, default=12.0)
    run.add_argument("--no-mlflow", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(func=cmd_run)

    wait = subparsers.add_parser("wait-leaderboard")
    wait.add_argument("--artifact", required=True)
    wait.set_defaults(func=cmd_wait)

    leaderboard = subparsers.add_parser("leaderboard")
    leaderboard.add_argument("--score", required=True, type=float)
    leaderboard.add_argument("--note", default="")
    leaderboard.set_defaults(func=cmd_leaderboard)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except CampaignStateError as exc:
        print(f"Campaign guard: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
