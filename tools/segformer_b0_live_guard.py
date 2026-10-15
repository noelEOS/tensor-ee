"""Exact-request guard for the separately approved SegFormer full-model run.

One allowlist covers SDK functions, Cloud methods, body fingerprints, assets and
recorded operation IDs. SDK/application retries are disabled. OAuth refresh and
connection-library physical sends are not measured by the application ledger.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
import weakref
from unittest import mock
from urllib.parse import unquote, urlparse, parse_qsl

import ee
from googleapiclient.http import HttpRequest
from googleapiclient.model import JsonModel

CAPS = {"batch": 16, "table": 9, "image": 7, "pixels": 204, "value": 16,
        "asset": 32, "algorithms": 2, "status": 480}
METHODS = {"earthengine.projects.table.export": "table", "earthengine.projects.image.export": "image",
           "earthengine.projects.image.computePixels": "pixels", "earthengine.projects.value.compute": "value",
           "earthengine.projects.assets.get": "asset", "earthengine.projects.algorithms.list": "algorithms",
           "earthengine.projects.operations.get": "status"}
PLUMBING = {"initialize", "is_initialized", "reset", "getAlgorithms", "getAsset", "getOperation",
            "exportImage", "exportTable", "getWorkloadTag", "setWorkloadTag", "resetWorkloadTag",
            "setDefaultWorkloadTag", "workloadTagContext", "setDeadline", "setMaxRetries", "getUserAgent",
            "setUserAgent", "setCloudApiUserProject", "setCloudApiKey", "authorizeHttp",
            "convert_asset_id_to_asset_name", "profiling"}


class RefusedLiveCall(RuntimeError):
    pass


class BudgetStop(RefusedLiveCall):
    pass


def wire(body):
    json.dumps(body, allow_nan=False)
    return JsonModel().serialize(body).encode("utf-8")


def fingerprint(body):
    raw = wire(body)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _resource(uri):
    path = unquote(urlparse(uri).path)
    match = re.search(r"(?:^|/)(projects/[^?]+)$", path)
    return match.group(1) if match else ""


@dataclass
class Guard:
    project: str
    approved: dict
    assets: set
    save: object = lambda record: None
    caps: dict = field(default_factory=lambda: dict(CAPS))
    counts: dict = field(default_factory=dict)
    ledger: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    operations: set = field(default_factory=set)
    used_bodies: set = field(default_factory=set)
    initializing: bool = False
    request_seconds: float = 300.0
    started: float = field(default_factory=time.monotonic)
    _authorized: object = field(default_factory=weakref.WeakSet)
    _executed: object = field(default_factory=weakref.WeakSet)

    def __post_init__(self):
        if set(self.caps) != set(CAPS) or any(type(v) is not int or not 0 <= v <= CAPS[k] for k, v in self.caps.items()):
            raise ValueError("guard caps may only lower the approved limits")
        if not self.assets or any(not a.startswith(f"projects/{self.project}/assets/") for a in self.assets):
            raise ValueError("guard requires explicit project-scoped asset destinations")
        self.counts = {k: 0 for k in CAPS} | self.counts

    def record(self):
        return {"counts": dict(self.counts), "sent": list(self.ledger), "refused": list(self.refused),
                "transport_behavior": {"sdk_num_retries": 0, "application_retries": 0,
                    "physical_http_attempts": "not instrumented; OAuth refresh and connection-library behavior may resend",
                    "ledger_scope": "authorized application Cloud requests; discovery/authentication are separate"}}

    def _stop(self, message, error=RefusedLiveCall):
        self.refused.append(message)
        self.save(self.record())
        raise error(message)

    def check_time(self):
        if time.monotonic() - self.started >= 7200:
            self._stop("two-hour global observation cap reached", BudgetStop)

    def add_operation(self, name):
        if not isinstance(name, str) or not re.fullmatch(re.escape(f"projects/{self.project}/operations/") + r"[A-Za-z0-9_-]+", name):
            self._stop("export response does not contain an exact operation name")
        self.operations.add(name)

    def after_call(self, result):
        try:
            self.check_time()
        except BudgetStop as error:
            error.returned_result = result
            raise
        return result

    def authorize(self, call):
        self.check_time()
        method = getattr(call, "methodId", None)
        kind = METHODS.get(method)
        if kind is None:
            self._stop(f"Cloud method {method!r} is outside the approved plan")
        uri = getattr(call, "uri", "")
        parsed = urlparse(uri)
        if parsed.scheme != "https" or parsed.hostname != "earthengine.googleapis.com" or any(
                (k, v) not in (("alt", "json"), ("prettyPrint", "false"), ("prettyPrint", "False"))
                for k, v in parse_qsl(parsed.query, keep_blank_values=True)):
            self._stop("Cloud URI host/query is outside the prepared endpoints")
        resource = _resource(uri)
        suffix = {"table": "table:export", "image": "image:export", "pixels": "image:computePixels",
                  "value": "value:compute", "algorithms": "algorithms"}.get(kind)
        if suffix and resource != f"projects/{self.project}/{suffix}":
            self._stop("Cloud resource does not match its approved method")
        if getattr(call, "method", None) != ("POST" if kind in ("table", "image", "pixels", "value") else "GET"):
            self._stop("HTTP verb does not match its approved method")
        if not resource.startswith(f"projects/{self.project}/"):
            self._stop("Cloud request project differs from the prepared plan")
        raw = getattr(call, "body", None)
        raw = raw.encode() if isinstance(raw, str) else (raw or b"")
        digest = hashlib.sha256(raw).hexdigest()
        label = resource
        if kind in ("table", "image", "pixels", "value"):
            expected = self.approved.get(kind, {}).get(digest)
            if expected is None:
                self._stop(f"unprepared or changed {kind} body")
            if digest in self.used_bodies:
                self._stop(f"{kind} body has already been attempted; no retry")
            cap = 8_000_000 if kind == "table" else 2_000_000
            if len(raw) > cap:
                self._stop(f"{kind} request exceeds its complete-body limit")
            label = expected["label"]
            if kind in ("table", "image"):
                body = json.loads(raw)
                destination_options = body.get("assetExportOptions", {}).get("earthEngineDestination", {})
                destination = destination_options.get("name")
                if destination not in self.assets or body.get("requestId") != expected["request_id"]:
                    self._stop("export destination/request UUID differs from its prepared envelope")
                if destination_options.get("overwrite") is not False:
                    self._stop("asset overwrite is forbidden")
                if self.counts["batch"] >= self.caps["batch"]:
                    self._stop("batch submission cap reached", BudgetStop)
        elif kind == "asset":
            if resource not in self.assets:
                self._stop("asset read is outside the explicit destination whitelist")
        elif kind == "status":
            if resource not in self.operations:
                self._stop("status read lacks a recorded exact operation ID")
        elif not self.initializing:
            self._stop("algorithms.list is only permitted during initialization")
        if self.counts[kind] >= self.caps[kind]:
            self._stop(f"{kind} call cap reached", BudgetStop)
        self.counts[kind] += 1
        if kind in ("table", "image"):
            self.counts["batch"] += 1
        if kind in ("table", "image", "pixels", "value"):
            self.used_bodies.add(digest)
        entry = {"method": method, "kind": kind, "label": label, "body_sha256": digest,
                 "body_bytes": len(raw), "unix": time.time(), "state": "ATTEMPTING"}
        self.ledger.append(entry)
        self._authorized.add(call)
        self.save(self.record())
        return entry

    @contextlib.contextmanager
    def installed(self):
        original_transport, original_execute = ee.data._execute_cloud_call, HttpRequest.execute

        def transport(call, num_retries=None):
            entry = self.authorize(call)
            begun = time.monotonic()
            try:
                result = original_transport(call, num_retries=0)
            except BaseException as error:
                entry.update(state="ERROR", error=f"{type(error).__name__}: {error}", wall_seconds=time.monotonic() - begun)
                self.save(self.record())
                raise
            entry.update(state="RETURNED", wall_seconds=time.monotonic() - begun)
            self.save(self.record())
            return self.after_call(result)

        def execute(call, *args, **kwargs):
            method = getattr(call, "methodId", None)
            if isinstance(method, str) and method.startswith("earthengine."):
                if call in self._executed:
                    self._stop("HttpRequest object already executed; no bypass retry")
                if call not in self._authorized:
                    self.authorize(call)
                self._executed.add(call)
            elif not self.initializing:
                self._stop("non-plan HttpRequest.execute outside initialization")
            kwargs["num_retries"] = 0
            # The pinned SDK uses an AuthorizedHttp around EE's requests-backed
            # _Http. Bound its timeout per request without rebuilding discovery.
            seconds = min(300.0, self.request_seconds, 7200 - (time.monotonic() - self.started))
            if seconds <= 0:
                self.check_time()
                self._stop("per-request observation deadline reached", BudgetStop)
            http = getattr(call, "http", None)
            transports = [http, getattr(http, "http", None), getattr(http, "_http", None)]
            changed = []
            for candidate in transports:
                for attribute in ("_timeout", "timeout"):
                    if candidate is not None and hasattr(candidate, attribute):
                        previous = getattr(candidate, attribute)
                        if previous is None or isinstance(previous, (int, float)):
                            setattr(candidate, attribute, min(seconds, previous) if previous and previous > 0 else seconds)
                            changed.append((candidate, attribute, previous))
            try:
                return self.after_call(original_execute(call, *args, **kwargs))
            finally:
                for candidate, attribute, previous in reversed(changed):
                    setattr(candidate, attribute, previous)

        def refuse(name):
            def rejected(*args, **kwargs):
                self._stop(f"ee.data.{name} is outside the approved function whitelist")
            return rejected

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ee.data, "_execute_cloud_call", transport))
            stack.enter_context(mock.patch.object(HttpRequest, "execute", execute))
            for name, value in vars(ee.data).copy().items():
                if not name.startswith("_") and callable(value) and not isinstance(value, type) and name not in PLUMBING:
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            yield self
