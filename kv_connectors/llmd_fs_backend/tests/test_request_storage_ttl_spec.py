# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
import pytest
from test_spec import make_spec
from vllm.v1.kv_offload.base import ReqContext

from llmd_fs_backend.manager import SharedStorageOffloadingManager
from llmd_fs_backend.request_manager import RequestStorageTTLManager

pytestmark = pytest.mark.no_cuda_required


@pytest.mark.parametrize("policy, prefix", [("fixed", "v1"), ("idle", "v3-idle")])
def test_allowlist_is_opt_in_with_explicit_retention_policy(tmp_path, policy, prefix):
    old = make_spec(tmp_path / "fixed", {"storage_ttl_seconds": 30})
    assert type(old.get_manager()) is SharedStorageOffloadingManager
    assert old.get_manager()._ttl_pins.idle
    assert "/.llmd-ttl-v3-idle/" in old.file_mapper.base_path
    new = make_spec(
        tmp_path / "mixed",
        {
            "storage_ttl_seconds": 30,
            "storage_ttl_allowlist": [1, 30],
            "storage_ttl_policy": policy,
        },
    )
    m = new.get_manager()
    assert isinstance(m, RequestStorageTTLManager) and new.get_manager() is m
    key = b"prefix" + (0).to_bytes(4, "big")
    c = ReqContext("r", {"llmd_storage_ttl_seconds": 1})
    m.on_new_request(c)
    assert m._tokens["r"].namespace == (
        ".llmd-ttl-v3-idle" if policy == "idle" else ".llmd-ttl-v1-1"
    )
    out = m.prepare_store([key], c)
    source, target = next(iter(out.store_spec.source_paths.items()))
    if policy == "idle":
        assert source == target
        assert "/.llmd-ttl-v3-idle/" in source
        assert out.store_spec.ttl_seconds == 1
    else:
        assert (
            f"/.llmd-ttl-{prefix}-30/" in source and f"/.llmd-ttl-{prefix}-1/" in target
        )
    assert source == new.file_mapper.get_file_name(key)
    m.complete_store([key], c, success=False)
    m.on_request_finished(c)
    m.shutdown()
    old.get_manager().shutdown()


@pytest.mark.parametrize(
    "extra",
    [
        {"storage_ttl_allowlist": [1, 30]},
        {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1]},
        {
            "storage_ttl_seconds": 30,
            "storage_ttl_allowlist": [1, 30],
            "enable_events": True,
        },
        {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1, 30], "backend": "OBJ"},
        {
            "storage_ttl_seconds": 30,
            "storage_ttl_allowlist": [1, 30],
            "gds_mode": "read_write",
        },
    ],
)
def test_spec_rejects_unsupported_allowlist_configuration(tmp_path, extra):
    with pytest.raises(ValueError):
        make_spec(tmp_path, extra)
