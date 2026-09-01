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
from src.experiment_config import load_profile  # noqa: E402


DEFAULT_STATE = ROOT / "data" / "experiments" / "campaign_state.json"
DEFAULT_EVENTS = ROOT / "data" / "experiments" / "campaign_events.jsonl"
DEFAULT_TELEGRAM_MARKER = (
    ROOT / "data" / "experiments" / "campaign_heartbeat_marker.json"
)
LOG_DIR = ROOT / "data" / "experiments" / "campaign_logs"
MLFLOW_SECRET_FILE = Path("/root/.iaaa_mlflow.env")
MLFLOW_SECRET_KEYS = {
    "DAGSHUB_USER_TOKEN",
    "DAGSHUB_REPO_OWNER",
    "DAGSHUB_REPO_NAME",
    "DAGSHUB_TRACKING_URI",
}


ANALYSIS_SPECS = {
    "frozen-eres2netv2-lme20-threefold-v1": {
        "config": Path(
            "configs/analyses/frozen-eres2netv2-lme20-threefold.json"
        ),
        "script": Path("scripts/audit_frozen_frontend_lme20.py"),
        "output": Path(
            "reports/generated/frozen_eres2netv2_lme20_threefold.json"
        ),
        "receipt_paths": (
            Path("scripts/audit_frozen_frontend_lme20.py"),
            Path(
                "reports/generated/"
                "FROZEN_ERES2NETV2_LME20_THREEFOLD_PREREGISTRATION_2026-08-31.md"
            ),
            Path(
                "data/experiments/frozen_eres2netv2_lme20/"
                "frozen_eres2netv2_embeddings.npz"
            ),
            Path(
                "data/experiments/frozen_eres2netv2_lme20/"
                "frozen_eres2netv2_embeddings.json"
            ),
            Path(
                "data/experiments/frozen_eres2netv2_lme20/"
                "frozen_eres2netv2_lme20_oof.npz"
            ),
            Path(
                "data/experiments/frozen_eres2netv2_lme20/"
                "frozen_eres2netv2_lme20_receipt.json"
            ),
        ),
    },
}


def _dotenv_value(raw_value: str) -> str:
    """Decode the common quoted-value subset without evaluating shell syntax."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def _notification_receipt(event_key: str) -> Optional[int]:
    if not DEFAULT_TELEGRAM_MARKER.is_file():
        return None
    marker = json.loads(DEFAULT_TELEGRAM_MARKER.read_text(encoding="utf-8"))
    event = (marker.get("events", {}) or {}).get(event_key, {}) or {}
    message_id = event.get("message_id")
    return message_id if isinstance(message_id, int) else None


def _record_notification(
    event_key: str,
    message_id: int,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    marker: dict[str, Any] = {}
    if DEFAULT_TELEGRAM_MARKER.is_file():
        marker = json.loads(DEFAULT_TELEGRAM_MARKER.read_text(encoding="utf-8"))
    sent_at = datetime.now(timezone.utc).isoformat()
    events = marker.setdefault("events", {})
    events[event_key] = {
        "message_id": int(message_id),
        "sent_at_utc": sent_at,
        "metadata": metadata or {},
    }
    marker.update({
        "last_event_key": event_key,
        "last_heartbeat_time": sent_at,
        "telegram_message_id": int(message_id),
    })
    DEFAULT_TELEGRAM_MARKER.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(marker, ensure_ascii=False, indent=2) + "\n"
    temporary = DEFAULT_TELEGRAM_MARKER.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, DEFAULT_TELEGRAM_MARKER)


def _notify(
    message: str,
    *,
    event_key: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    try:
        from telegram_notifier import send

        if event_key:
            existing = _notification_receipt(event_key)
            if existing is not None:
                return existing
        message_id = send(message)
        if not isinstance(message_id, int):
            raise RuntimeError("Telegram notifier returned no integer message_id")
        if event_key:
            _record_notification(event_key, message_id, metadata)
        return message_id
    except Exception as exc:
        # Notification is best-effort and must never kill a training run.
        print(f"Telegram notification warning: {type(exc).__name__}", flush=True)
        return None


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


def cmd_set_max_campaign_cost(args: argparse.Namespace) -> int:
    state = _store(args).set_max_campaign_cost(args.cost_usd, args.reason)
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
    config = load_profile(args.profile)
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
    experiment = config.get("experiment", {}) or {}
    model = config.get("model", {}) or {}
    split = (config.get("data", {}) or {}).get("split", {}) or {}
    training = config.get("training", {}) or {}
    gate = experiment.get("preregistered_gate", {}) or {}
    _notify(
        "▶️ آزمایش جدید آغاز شد\n\n"
        f"پروفایل: {args.profile}\n"
        f"مرحله: {args.stage}\n"
        f"نسخهٔ کد: {commit[:8]}\n"
        f"Encoder: {model.get('encoder_type', 'نامشخص')}\n"
        f"Split: fold={split.get('fold')}/{split.get('folds')}، seed={split.get('seed')}\n"
        f"افق: حداکثر {training.get('epochs')} ایپاک با early stopping "
        f"از {training.get('early_stopping_start_epoch')} و patience="
        f"{training.get('early_stopping_patience')}\n"
        f"سقف زمان: {args.timeout_hours:g} ساعت\n\n"
        "🧠 تفسیر علمی\n"
        f"فرضیه: {experiment.get('purpose', 'آزمایش ازپیش‌ثبت‌شده')}\n"
        "تصمیم فقط Raw probability-average + argmax است؛ EMA/logit و "
        "LME20 نقش diagnostic دارند. هیچ threshold یا blend از OOF/leaderboard "
        "تنظیم نمی‌شود.\n"
        f"Gate مستقل: {gate.get('standalone_min_raw_macro_f1', 'ثبت نشده')}؛ "
        f"gain فیوژن ثابت: {gate.get('fixed_50_50_min_macro_gain', 'ثبت نشده')}.\n"
        "شروع Run به‌معنای تأیید candidate نیست؛ نتیجه فقط پس از audit artifact "
        "و guardrail هم‌زمان Known/OOD پذیرفته می‌شود.",
        event_key=f"{args.profile}:started:{config_sha}",
        metadata={
            "profile": args.profile,
            "git_commit": commit,
            "config_sha256": config_sha,
            "stage": args.stage,
        },
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
            "وضعیت: تحلیل عمیق نتیجه\n\n"
            "🧠 تفسیر علمی\n"
            "کامل‌شدن process فقط سلامت اجرایی را نشان می‌دهد؛ هنوز هیچ نتیجه‌ای "
            "پذیرفته نشده و OOF، منحنی کامل، Known/OOD، rescue و hashها باید "
            "طبق gate ازپیش‌ثبت‌شده audit شوند.",
            event_key=f"{args.profile}:finished:{config_sha}:0",
            metadata={
                "profile": args.profile,
                "git_commit": commit,
                "config_sha256": config_sha,
                "exit_code": 0,
            },
        )
    else:
        _notify(
            "⛔ آزمایش متوقف شد\n\n"
            f"پروفایل: {args.profile}\n"
            f"کد خروج: {exit_code}\n"
            f"علت: {telegram_reason}\n"
            f"گزارش: {log_path.relative_to(ROOT)}\n\n"
            "هیچ اجرای بعدی تا تحلیل علت آغاز نمی‌شود.\n\n"
            "🧠 تفسیر علمی\n"
            "این رخداد به‌تنهایی شکست فرضیه نیست؛ ابتدا باید مشخص شود توقف ناشی "
            "از futility علمی، timeout، OOM یا خطای provenance بوده است.",
            event_key=f"{args.profile}:finished:{config_sha}:{exit_code}",
            metadata={
                "profile": args.profile,
                "git_commit": commit,
                "config_sha256": config_sha,
                "exit_code": exit_code,
            },
        )
    print(json.dumps(_status_payload(store), indent=2, ensure_ascii=False))
    return exit_code


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run one allowlisted preregistered analysis under campaign guards."""
    store = _store(args)
    state = store.load()
    if state["status"] not in {"READY", "ANALYZING"}:
        raise CampaignStateError(
            f"cannot start an analysis from state {state['status']}"
        )
    spec = ANALYSIS_SPECS.get(args.analysis)
    if spec is None:
        raise CampaignStateError(f"unknown preregistered analysis: {args.analysis}")
    commit = _assert_reproducible_checkout(args.allow_dirty)

    config_path = ROOT / spec["config"]
    script_path = ROOT / spec["script"]
    output_path = ROOT / spec["output"]
    if not config_path.is_file() or not script_path.is_file():
        raise CampaignStateError(
            f"analysis inputs are incomplete for {args.analysis}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("analysis_id") != args.analysis:
        raise CampaignStateError("analysis id does not match its locked config")
    timeout_hours = float(config["runtime"]["timeout_hours"])
    maximum_cost = float(config["runtime"]["maximum_incremental_cost_usd"])
    if timeout_hours <= 0.0 or maximum_cost <= 0.0:
        raise CampaignStateError("analysis runtime guard must be positive")
    projected_cost = timeout_hours * float(state["policy"]["hourly_rate_usd"])
    if projected_cost > maximum_cost + 1e-12:
        raise CampaignStateError(
            "analysis preregistration cost guard rejected the operation: "
            f"projected=${projected_cost:.3f}, locked=${maximum_cost:.3f}"
        )
    reserve_hours = min(timeout_hours, state["policy"]["max_run_hours"])
    store.assert_budget(reserve_hours=reserve_hours)

    config_sha = _sha256(config_path)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"{stamp}_{args.analysis}.log"
    command = [
        sys.executable, "-X", "utf8", str(script_path),
        "--analysis-config", str(config_path),
        "--output", str(output_path),
    ]
    run_record = {
        "profile": args.analysis,
        "stage": "analysis",
        "git_commit": commit,
        "config_sha256": config_sha,
        "log_path": str(log_path.relative_to(ROOT)),
        "command": command,
    }
    store.start_run(run_record)
    _notify(
        "🔬 تحلیل ازپیش‌ثبت‌شده آغاز شد\n\n"
        f"شناسه: {args.analysis}\n"
        f"نسخهٔ کد: {commit[:8]}\n"
        f"سقف زمان: {timeout_hours:g} ساعت\n"
        f"سقف هزینهٔ افزوده: ${maximum_cost:.2f}\n\n"
        "هیچ پارامتر یا variant بر اساس نتیجه انتخاب نخواهد شد."
    )

    timeout_seconds = 3600 * min(
        timeout_hours, float(state["policy"]["max_run_hours"]),
    )
    exit_code = 1
    reason = "analysis failed before process start"
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
            f"analysis {args.analysis} completed"
            if exit_code == 0
            else f"analysis {args.analysis} failed with exit code {exit_code}"
        )
    except subprocess.TimeoutExpired:
        exit_code = 124
        reason = f"analysis {args.analysis} exceeded its time limit"

    extra_paths = (
        config_path,
        log_path,
        output_path,
        *(ROOT / path for path in spec["receipt_paths"]),
    )
    artifacts = _artifact_receipts(args.analysis, extra_paths=extra_paths)
    success = exit_code == 0
    final_state = store.finish_run(
        success=success,
        exit_code=exit_code,
        reason=reason,
        artifacts=artifacts,
    )
    budget = store.budget_snapshot(final_state)
    if success:
        _notify(
            "✅ تحلیل frozen ERes2NetV2 کامل شد\n\n"
            f"شناسه: {args.analysis}\n"
            f"artifactهای receipt: {len(artifacts)}\n"
            f"هزینهٔ تخمینی کمپین: ${budget['estimated_cost_usd']:.2f}\n\n"
            "نتیجه فقط طبق gate ازپیش‌ثبت‌شده تفسیر می‌شود."
        )
    else:
        _notify(
            "⛔ تحلیل frozen ERes2NetV2 متوقف شد\n\n"
            f"شناسه: {args.analysis}\n"
            f"کد خروج: {exit_code}\n"
            f"گزارش: {log_path.relative_to(ROOT)}\n\n"
            "هیچ اجرای جایگزینی خودکار آغاز نشده است."
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

    max_campaign_cost = subparsers.add_parser("set-max-campaign-cost")
    max_campaign_cost.add_argument("--cost-usd", required=True, type=float)
    max_campaign_cost.add_argument("--reason", required=True)
    max_campaign_cost.set_defaults(func=cmd_set_max_campaign_cost)

    run = subparsers.add_parser("run")
    run.add_argument("--profile", required=True)
    run.add_argument("--stage", choices=["all", "data", "train", "eval"], default="eval")
    run.add_argument("--timeout-hours", type=float, default=12.0)
    run.add_argument("--no-mlflow", action="store_true")
    run.add_argument("--allow-dirty", action="store_true")
    run.set_defaults(func=cmd_run)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--analysis", required=True, choices=sorted(ANALYSIS_SPECS))
    analyze.add_argument("--allow-dirty", action="store_true")
    analyze.set_defaults(func=cmd_analyze)

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
