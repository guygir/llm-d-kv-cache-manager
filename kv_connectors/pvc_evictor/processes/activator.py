"""Activator process for monitoring disk usage and controlling deletion."""

import logging
import multiprocessing
import time

from utils.logging_helpers import send_stats_to_queue
from utils.system import get_disk_usage_from_statvfs, setup_logging


def activator_process(
    process_num: int,
    mount_path: str,
    cleanup_threshold: float,
    target_threshold: float,
    logger_interval: float,
    deletion_event: multiprocessing.Event,
    result_queue: multiprocessing.Queue,
    shutdown_event: multiprocessing.Event,
    config_dict: dict,
):
    """
    Activator process (P(N+1)): Monitors disk usage and controls deletion trigger.

    Monitors statvfs() every logger_interval seconds, sets deletion_event when
    usage > cleanup_threshold, and clears deletion_event when usage < target_threshold.

    Reports statistics to main process for aggregated logging.
    """
    log_file = config_dict.get("log_file_path")
    setup_logging(config_dict.get("log_level", "INFO"), process_num, log_file)
    logger = logging.getLogger("activator")

    # Log activator startup information
    logger.info(f"Activator P{process_num} started - monitoring every {logger_interval}s")
    logger.info(f"Activator P{process_num} thresholds: cleanup={cleanup_threshold}%, target={target_threshold}%")

    controller = None
    sampler = None
    if config_dict.get("cleanup_policy"):
        from cleanup_policy import CleanupController, CleanupSettings

        settings = CleanupSettings.from_dict(config_dict["cleanup_policy"])
        controller = CleanupController(settings, cleanup_threshold, target_threshold)
        from activity import ActivitySampler

        sampler = ActivitySampler(settings.sources, settings.poll_timeout_seconds)
    deletion_active = False
    last_stats_send_time = 0.0

    try:
        while not shutdown_event.is_set():
            usage = get_disk_usage_from_statvfs(mount_path)

            if controller is not None:
                from activity import Activity

                needs_sample = usage is not None and (controller.requested or usage.usage_percent >= cleanup_threshold)
                activity = sampler.sample() if needs_sample else Activity(None, "no actionable cleanup pressure")
                decision = controller.decide(usage.usage_percent if usage else None, activity.idle, time.monotonic())
                if decision.enabled:
                    deletion_event.set()
                else:
                    deletion_event.clear()
                if decision != getattr(controller, "last_decision", None):
                    logger.info(
                        "Cleanup mode=%s admitted=%s reason=%s activity=%s",
                        decision.mode,
                        decision.enabled,
                        decision.reason,
                        activity.reason,
                    )
                    controller.last_decision = decision
                if usage:
                    last_stats_send_time = send_stats_to_queue(
                        result_queue,
                        "activator_stats",
                        process_num,
                        {
                            "usage_percent": usage.usage_percent,
                            "used_bytes": usage.used_bytes,
                            "available_bytes": usage.available_bytes,
                            "total_bytes": usage.total_bytes,
                            "deletion_active": decision.enabled,
                            "cleanup_mode": decision.mode,
                            "activity_known": activity.idle is not None,
                        },
                        last_stats_send_time,
                    )
                shutdown_event.wait(logger_interval)
                continue

            if usage:
                current_time = time.time()

                # Send stats periodically to main process for aggregated logging
                last_stats_send_time = send_stats_to_queue(
                    result_queue,
                    "activator_stats",
                    process_num,
                    {
                        "usage_percent": usage.usage_percent,
                        "used_bytes": usage.used_bytes,
                        "available_bytes": usage.available_bytes,
                        "total_bytes": usage.total_bytes,
                        "deletion_active": deletion_active,
                    },
                    last_stats_send_time,
                )

                # Control deletion based on thresholds (always log these critical events)
                # Trigger deletion when usage exceeds cleanup threshold (hysteresis: ON at cleanup%, OFF at target%)
                if usage.usage_percent >= cleanup_threshold and not deletion_active:
                    # Log deletion activation (always immediate, even with aggregated logging)
                    logger.warning(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% >= "
                        f"{cleanup_threshold}% - Triggering deletion ON"
                    )
                    logger.info(
                        f"DELETION_START: timestamp={current_time:.3f}, usage={usage.usage_percent:.2f}%, "
                        f"used={usage.used_bytes / (1024**3):.2f}GB, "
                        f"usable={(usage.used_bytes + usage.available_bytes) / (1024**3):.2f}GB, "
                        f"total={usage.total_bytes / (1024**3):.2f}GB"
                    )
                    deletion_event.set()
                    deletion_active = True
                elif usage.usage_percent <= target_threshold and deletion_active:
                    # Log deletion deactivation (always immediate, even with aggregated logging)
                    logger.info(
                        f"Activator P{process_num}: Usage {usage.usage_percent:.2f}% <= "
                        f"{target_threshold}% - Triggering deletion OFF"
                    )
                    logger.info(
                        f"DELETION_END: timestamp={current_time:.3f}, usage={usage.usage_percent:.2f}%, "
                        f"used={usage.used_bytes / (1024**3):.2f}GB, "
                        f"usable={(usage.used_bytes + usage.available_bytes) / (1024**3):.2f}GB, "
                        f"total={usage.total_bytes / (1024**3):.2f}GB"
                    )
                    deletion_event.clear()
                    deletion_active = False

            time.sleep(logger_interval)

    except Exception as e:
        logger.exception(f"Activator P{process_num} error: {e}")
    finally:
        if sampler is not None:
            sampler.close()
        logger.info(f"Activator P{process_num} stopping")
        deletion_event.clear()
