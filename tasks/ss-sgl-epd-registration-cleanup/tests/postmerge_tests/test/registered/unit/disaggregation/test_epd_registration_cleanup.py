# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def _bootstrap_server():
    from sglang.srt.disaggregation.encode_receiver import EncoderBootstrapServer

    with patch.object(EncoderBootstrapServer, "_run_server"):
        server = EncoderBootstrapServer(
            "127.0.0.1",
            0,
            health_check_interval=0,
        )
        server.thread.join(timeout=5)
    return server


def _server_args(**overrides):
    values = {
        "host": "0.0.0.0",
        "port": 31000,
        "ssl_certfile": None,
        "encoder_register_urls": [
            "http://bootstrap-a:8997",
            "http://bootstrap-b:8997",
        ],
        "dp_size": 1,
        "tp_size": 1,
        "dist_init_addr": None,
        "base_gpu_id": 0,
        "url": lambda: "http://127.0.0.1:31000",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _invoke_installed_callbacks(install_cleanup):
    for installed in install_cleanup.call_args_list:
        callback, *args = installed.args
        callback(*args, **installed.kwargs)


class TestRegistrationCleanup(unittest.TestCase):
    def test_reregister_and_unregister_clear_stale_health_failures(self):
        url = "http://encoder:31000"
        server = _bootstrap_server()

        self.assertTrue(server.register(url))
        server._consecutive_failures[url] = 2
        self.assertFalse(server.register(url))
        self.assertNotIn(url, server._consecutive_failures)

        server._consecutive_failures[url] = 2
        self.assertTrue(server.unregister(url))
        self.assertNotIn(url, server._consecutive_failures)
        self.assertNotIn(url, server.list_urls())

    def test_registration_advertises_routable_tls_aware_url(self):
        from sglang.srt.disaggregation import encode_server

        cases = (
            ("0.0.0.0", "10.2.3.4", True, "https://10.2.3.4:31000"),
            ("::", "2001:db8::4", True, "https://[2001:db8::4]:31000"),
            (None, "10.2.3.5", True, "https://10.2.3.5:31000"),
            ("encoder.internal", "unused", False, "https://encoder.internal:31000"),
        )
        response = Mock(status_code=200, text="OK")
        for host, resolved, resolves_host, expected_url in cases:
            with self.subTest(host=host):
                args = _server_args(host=host, ssl_certfile="/tmp/server.crt")
                with (
                    patch.object(encode_server, "get_local_ip_auto", return_value=resolved) as resolve,
                    patch.object(encode_server.http_requests, "post", return_value=response) as post,
                    patch.object(encode_server.threading, "Thread", _ImmediateThread),
                ):
                    encode_server._register_encoder_url_with_bootstrap(args)

                expected_payload = {"url": expected_url}
                self.assertEqual(post.call_count, 2)
                post.assert_has_calls(
                    [
                        call(
                            "http://bootstrap-a:8997/register_encoder_url",
                            json=expected_payload,
                            timeout=5.0,
                        ),
                        call(
                            "http://bootstrap-b:8997/register_encoder_url",
                            json=expected_payload,
                            timeout=5.0,
                        ),
                    ]
                )
                if resolves_host:
                    resolve.assert_called_once_with(host)
                else:
                    resolve.assert_not_called()

    def test_unregister_notifies_every_bootstrap_with_matching_url(self):
        from sglang.srt.disaggregation import encode_server

        cases = (
            ("0.0.0.0", "10.2.3.4", "https://10.2.3.4:31000"),
            ("::", "2001:db8::4", "https://[2001:db8::4]:31000"),
            (None, "10.2.3.5", "https://10.2.3.5:31000"),
            ("encoder.internal", "unused", "https://encoder.internal:31000"),
        )
        response = Mock(status_code=200, text="OK")

        class _PortArgs:
            nccl_port = 32100

        for host, resolved, expected_url in cases:
            with self.subTest(host=host):
                args = _server_args(host=host, ssl_certfile="/tmp/server.crt")
                fake_context = Mock()
                fake_context.Process.return_value = Mock()
                with (
                    patch.object(encode_server, "configure_logger"),
                    patch.object(encode_server.mp, "get_context", return_value=fake_context),
                    patch.object(encode_server.zmq, "Context", return_value=Mock()),
                    patch.object(encode_server.PortArgs, "init_new", return_value=_PortArgs()),
                    patch.object(encode_server, "MMEncoder", return_value=Mock()),
                    patch.object(
                        encode_server,
                        "get_local_ip_auto",
                        side_effect=(resolved, "203.0.113.99"),
                    ) as resolve,
                    patch.object(encode_server.http_requests, "post", return_value=response) as post,
                    patch.object(encode_server.http_requests, "delete", return_value=response) as delete,
                    patch.object(encode_server.threading, "Thread", _ImmediateThread),
                    patch("atexit.register") as install_cleanup,
                    patch.object(encode_server.uvicorn, "run"),
                ):
                    encode_server.launch_server(args)
                    args.host = "changed-after-registration.invalid"
                    args.port = 32000
                    args.ssl_certfile = None
                    args.encoder_register_urls = ["http://changed-bootstrap:8997"]
                    _invoke_installed_callbacks(install_cleanup)

                expected_payload = {"url": expected_url}
                self.assertEqual(
                    [request.kwargs["json"] for request in post.call_args_list],
                    [expected_payload, expected_payload],
                )
                delete.assert_has_calls(
                    [
                        call(
                            "http://bootstrap-a:8997/unregister_encoder_url",
                            json=expected_payload,
                            timeout=2.0,
                        ),
                        call(
                            "http://bootstrap-b:8997/unregister_encoder_url",
                            json=expected_payload,
                            timeout=2.0,
                        ),
                    ]
                )
                self.assertEqual(
                    [request.kwargs["json"] for request in delete.call_args_list],
                    [expected_payload, expected_payload],
                )
                expected_resolutions = 0 if host == "encoder.internal" else 1
                self.assertEqual(resolve.call_count, expected_resolutions)

    def test_unregister_continues_after_exception_and_non_200(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(
            host="encoder.internal",
            encoder_register_urls=[
                "http://bootstrap-exception:8997",
                "http://bootstrap-rejected:8997",
                "http://bootstrap-ok:8997",
            ],
        )

        class _PortArgs:
            nccl_port = 32100

        fake_context = Mock()
        fake_context.Process.return_value = Mock()
        response = Mock(status_code=200, text="OK")
        with (
            patch.object(encode_server, "configure_logger"),
            patch.object(encode_server.mp, "get_context", return_value=fake_context),
            patch.object(encode_server.zmq, "Context", return_value=Mock()),
            patch.object(encode_server.PortArgs, "init_new", return_value=_PortArgs()),
            patch.object(encode_server, "MMEncoder", return_value=Mock()),
            patch.object(encode_server.http_requests, "post", return_value=response),
            patch.object(
                encode_server.http_requests,
                "delete",
                side_effect=(
                    RuntimeError("offline"),
                    Mock(status_code=503, text="busy"),
                    Mock(status_code=200, text="OK"),
                ),
            ) as delete,
            patch.object(encode_server.threading, "Thread", _ImmediateThread),
            patch("atexit.register") as install_cleanup,
            patch.object(encode_server.uvicorn, "run"),
        ):
            encode_server.launch_server(args)
            _invoke_installed_callbacks(install_cleanup)

        expected_payload = {"url": "http://encoder.internal:31000"}
        self.assertEqual(
            delete.call_args_list,
            [
                call(
                    "http://bootstrap-exception:8997/unregister_encoder_url",
                    json=expected_payload,
                    timeout=2.0,
                ),
                call(
                    "http://bootstrap-rejected:8997/unregister_encoder_url",
                    json=expected_payload,
                    timeout=2.0,
                ),
                call(
                    "http://bootstrap-ok:8997/unregister_encoder_url",
                    json=expected_payload,
                    timeout=2.0,
                ),
            ],
        )

    def test_single_and_dp_launch_install_shutdown_cleanup(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(host="encoder.internal", dp_size=2)
        fake_context = Mock()
        fake_processes = [Mock(), Mock()]
        for process in fake_processes:
            process.is_alive.return_value = False
        fake_context.Process.side_effect = fake_processes
        fake_zmq_context = Mock()
        response = Mock(status_code=200, text="OK")

        def fake_reindex(_gpu_id):
            return contextlib.nullcontext(0)

        with (
            patch.object(encode_server.mp, "get_context", return_value=fake_context),
            patch.object(encode_server.zmq.asyncio, "Context", return_value=fake_zmq_context),
            patch.object(encode_server, "get_zmq_socket", return_value=Mock()),
            patch.object(encode_server, "maybe_reindex_device_id", fake_reindex),
            patch.object(encode_server, "DPDispatcher", return_value=Mock()),
            patch.object(encode_server.http_requests, "post", return_value=response) as post,
            patch.object(encode_server.http_requests, "delete", return_value=response) as delete,
            patch.object(encode_server.threading, "Thread", _ImmediateThread),
            patch("atexit.register") as install_cleanup,
            patch.object(encode_server.uvicorn, "run"),
        ):
            encode_server._launch_server_dp(args)
            args.host = "changed-after-registration.invalid"
            args.port = 32000
            args.ssl_certfile = "/tmp/changed.crt"
            args.encoder_register_urls = ["http://changed-bootstrap:8997"]
            _invoke_installed_callbacks(install_cleanup)

        expected_payload = {"url": "http://encoder.internal:31000"}
        self.assertEqual([request.kwargs["json"] for request in post.call_args_list], [expected_payload] * 2)
        self.assertEqual(
            [request.kwargs["json"] for request in delete.call_args_list],
            [expected_payload] * 2,
        )


class TestRegistrationRetained(unittest.TestCase):
    def test_each_bootstrap_retries_independently(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(
            host="encoder.internal",
            url=lambda: "http://encoder.internal:31000",
        )
        responses = (
            Mock(status_code=503, text="busy"),
            Mock(status_code=200, text="OK"),
            Mock(status_code=200, text="OK"),
        )
        with (
            patch.object(encode_server.http_requests, "post", side_effect=responses) as post,
            patch.object(encode_server.threading, "Thread", _ImmediateThread),
            patch.object(encode_server.time, "sleep") as sleep,
        ):
            encode_server._register_encoder_url_with_bootstrap(args)

        payload = {"url": "http://encoder.internal:31000"}
        self.assertEqual(
            post.call_args_list,
            [
                call(
                    "http://bootstrap-a:8997/register_encoder_url",
                    json=payload,
                    timeout=5.0,
                ),
                call(
                    "http://bootstrap-b:8997/register_encoder_url",
                    json=payload,
                    timeout=5.0,
                ),
                call(
                    "http://bootstrap-a:8997/register_encoder_url",
                    json=payload,
                    timeout=5.0,
                ),
            ],
        )
        sleep.assert_called_once_with(5.0)

    def test_registration_retries_after_exception(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(
            host="encoder.internal",
            encoder_register_urls=["http://bootstrap-a:8997"],
            url=lambda: "http://encoder.internal:31000",
        )
        with (
            patch.object(
                encode_server.http_requests,
                "post",
                side_effect=(RuntimeError("offline"), Mock(status_code=200, text="OK")),
            ) as post,
            patch.object(encode_server.threading, "Thread", _ImmediateThread),
            patch.object(encode_server.time, "sleep") as sleep,
        ):
            encode_server._register_encoder_url_with_bootstrap(args)

        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(5.0)

    def test_registration_stops_at_retry_limit(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(
            host="encoder.internal",
            encoder_register_urls=["http://bootstrap-a:8997"],
            url=lambda: "http://encoder.internal:31000",
        )
        response = Mock(status_code=503, text="busy")
        with (
            patch.object(encode_server.http_requests, "post", return_value=response) as post,
            patch.object(encode_server.threading, "Thread", _ImmediateThread),
            patch.object(encode_server.time, "sleep") as sleep,
        ):
            encode_server._register_encoder_url_with_bootstrap(args)

        self.assertEqual(post.call_count, 30)
        self.assertEqual(sleep.call_count, 29)
        sleep.assert_has_calls([call(5.0)] * 29)

    def test_duplicate_registration_remains_idempotent(self):
        url = "http://encoder:31000"
        server = _bootstrap_server()
        self.assertTrue(server.register(url))
        self.assertFalse(server.register(url))
        self.assertEqual(server.list_urls(), [url])

    def test_unknown_unregister_remains_noop(self):
        server = _bootstrap_server()
        self.assertFalse(server.unregister("http://missing:31000"))
        self.assertEqual(server.list_urls(), [])

    def test_empty_bootstrap_target_list_remains_noop(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(encoder_register_urls=[], host="127.0.0.1")
        with (
            patch.object(encode_server.http_requests, "post") as post,
            patch.object(encode_server.threading, "Thread") as thread,
        ):
            registration = encode_server._register_encoder_url_with_bootstrap(args)
        self.assertIsNone(registration)
        post.assert_not_called()
        thread.assert_not_called()

    def test_launch_without_bootstrap_does_not_install_cleanup(self):
        from sglang.srt.disaggregation import encode_server

        args = _server_args(encoder_register_urls=[])

        class _PortArgs:
            nccl_port = 32100

        fake_context = Mock()
        fake_context.Process.return_value = Mock()
        with (
            patch.object(encode_server, "configure_logger"),
            patch.object(encode_server.mp, "get_context", return_value=fake_context),
            patch.object(encode_server.zmq, "Context", return_value=Mock()),
            patch.object(encode_server.PortArgs, "init_new", return_value=_PortArgs()),
            patch.object(encode_server, "MMEncoder", return_value=Mock()),
            patch.object(encode_server, "_register_encoder_url_with_bootstrap") as register,
            patch("atexit.register") as install_cleanup,
            patch.object(encode_server.uvicorn, "run"),
        ):
            encode_server.launch_server(args)

        register.assert_not_called()
        install_cleanup.assert_not_called()

        args.dp_size = 2
        fake_processes = [Mock(), Mock()]
        for process in fake_processes:
            process.is_alive.return_value = False
        fake_context.Process.side_effect = fake_processes

        def fake_reindex(_gpu_id):
            return contextlib.nullcontext(0)

        with (
            patch.object(encode_server.mp, "get_context", return_value=fake_context),
            patch.object(encode_server.zmq.asyncio, "Context", return_value=Mock()),
            patch.object(encode_server, "get_zmq_socket", return_value=Mock()),
            patch.object(encode_server, "maybe_reindex_device_id", fake_reindex),
            patch.object(encode_server, "DPDispatcher", return_value=Mock()),
            patch.object(encode_server, "_register_encoder_url_with_bootstrap") as register,
            patch.object(encode_server.http_requests, "delete") as delete,
            patch("atexit.register") as install_cleanup,
            patch.object(encode_server.uvicorn, "run"),
        ):
            encode_server._launch_server_dp(args)
            _invoke_installed_callbacks(install_cleanup)

        register.assert_not_called()
        delete.assert_not_called()
