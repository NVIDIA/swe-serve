# Copyright 2023-2024 SGLang Team
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the speculative algorithm plugin registry."""

import itertools
import unittest
from unittest.mock import MagicMock

_VerifierTestCase = unittest.TestCase

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

_NAME_COUNTER = itertools.count()


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{next(_NAME_COUNTER)}"


class _RegistryIsolated(_VerifierTestCase):
    """Use unique plugin names so tests don't depend on private registry state."""


class TestFromString(_RegistryIsolated):
    def test_none_input_returns_none_member(self):
        self.assertIs(SpeculativeAlgorithm.from_string(None), SpeculativeAlgorithm.NONE)

    def test_builtin_name_returns_enum(self):
        self.assertIs(SpeculativeAlgorithm.from_string("EAGLE"), SpeculativeAlgorithm.EAGLE)
        self.assertIs(SpeculativeAlgorithm.from_string("NGRAM"), SpeculativeAlgorithm.NGRAM)

    def test_builtin_name_is_case_insensitive(self):
        self.assertIs(SpeculativeAlgorithm.from_string("eagle"), SpeculativeAlgorithm.EAGLE)

    def test_unknown_name_raises(self):
        name = _unique_name("NOT_REGISTERED")
        with self.assertRaises(ValueError) as raised:
            SpeculativeAlgorithm.from_string(name)
        self.assertIn(name, str(raised.exception))

    def test_registered_plugin_returns_custom_spec(self):
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string(name)
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertEqual(algo.name, name)

    def test_registered_plugin_lookup_is_case_insensitive(self):
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name)
        def _factory(server_args):
            return MagicMock

        self.assertIs(
            SpeculativeAlgorithm.from_string(name.lower()),
            SpeculativeAlgorithm.from_string(name),
        )


class TestRegister(_RegistryIsolated):
    def test_register_returns_factory_unchanged(self):
        name = _unique_name("MY_FOO")

        def _factory(server_args):
            return MagicMock

        decorated = SpeculativeAlgorithm.register(name)(_factory)
        self.assertIs(decorated, _factory)

    def test_two_distinct_registrations_are_independent(self):
        foo_name = _unique_name("FOO")
        bar_name = _unique_name("BAR")

        @SpeculativeAlgorithm.register(foo_name)
        def _foo_factory(server_args):
            return MagicMock

        @SpeculativeAlgorithm.register(bar_name)
        def _bar_factory(server_args):
            return MagicMock

        foo = SpeculativeAlgorithm.from_string(foo_name)
        bar = SpeculativeAlgorithm.from_string(bar_name)
        self.assertIsNot(foo, bar)
        self.assertNotEqual(foo, bar)
        self.assertEqual(foo.name, foo_name)
        self.assertEqual(bar.name, bar_name)

    def test_duplicate_name_raises(self):
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name)
        def _factory(server_args):
            return MagicMock

        with self.assertRaises(ValueError):

            @SpeculativeAlgorithm.register(name)
            def _factory2(server_args):
                return MagicMock

    def test_reserved_name_raises(self):
        reserved_names = [
            "DFLASH",
            "EAGLE",
            "EAGLE3",
            "NEXTN",
            "STANDALONE",
            "NGRAM",
            "NONE",
        ]

        def _factory(server_args):
            return MagicMock

        for reserved in reserved_names:
            with self.assertRaises(ValueError):
                SpeculativeAlgorithm.register(reserved)(_factory)

    def test_register_is_case_insensitive_on_collision(self):
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name)
        def _factory(server_args):
            return MagicMock

        with self.assertRaises(ValueError):

            @SpeculativeAlgorithm.register(name.lower())
            def _factory2(server_args):
                return MagicMock


class TestCustomSpecAlgoInterface(_RegistryIsolated):
    """CustomSpecAlgo must duck-type SpeculativeAlgorithm enum values."""

    def setUp(self):
        super().setUp()
        self.name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(self.name, supports_overlap=False)
        def _factory(server_args):
            return MagicMock

        self.algo = SpeculativeAlgorithm.from_string(self.name)

    def test_is_predicates_all_false_except_speculative(self):
        self.assertFalse(self.algo.is_none())
        self.assertFalse(self.algo.is_eagle())
        self.assertFalse(self.algo.is_eagle3())
        self.assertFalse(self.algo.is_dflash())
        self.assertFalse(self.algo.is_standalone())
        self.assertFalse(self.algo.is_ngram())
        self.assertTrue(self.algo.is_speculative())

    def test_supports_spec_v2_follows_supports_overlap(self):
        # Plugin registered with supports_overlap=False -> not spec_v2.
        self.assertFalse(self.algo.supports_spec_v2())

        name = _unique_name("MY_V2")

        @SpeculativeAlgorithm.register(name, supports_overlap=True)
        def _factory(server_args):
            return MagicMock

        v2 = SpeculativeAlgorithm.from_string(name)
        self.assertTrue(v2.supports_spec_v2())

    def test_create_worker_calls_factory(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = True
        worker_cls = self.algo.create_worker(server_args)
        self.assertIs(worker_cls, MagicMock)

    def test_create_worker_raises_on_overlap_mismatch(self):
        server_args = MagicMock()
        server_args.disable_overlap_schedule = False
        with self.assertRaises(ValueError):
            self.algo.create_worker(server_args)


class TestValidatorHook(_RegistryIsolated):
    def test_validator_invocation_is_caller_driven(self):
        validator = MagicMock()
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name, validate_server_args=validator)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string(name)
        self.assertIs(algo.validate_server_args, validator)
        # Callers (e.g. ServerArgs.__post_init__) must invoke the hook themselves;
        # CustomSpecAlgo does not call it from create_worker.
        validator.assert_not_called()


class TestSubclassOverride(_RegistryIsolated):
    """Plugins can subclass CustomSpecAlgo to override is_*() / create_worker."""

    def test_subclass_overrides_is_eagle(self):
        name = _unique_name("MY_LIKE_EAGLE")

        class EagleLike(CustomSpecAlgo):
            def is_eagle(self) -> bool:
                return True

        @SpeculativeAlgorithm.register(name, supports_overlap=True, spec_class=EagleLike)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string(name)
        self.assertIsInstance(algo, EagleLike)
        self.assertIsInstance(algo, CustomSpecAlgo)
        self.assertTrue(algo.is_eagle())
        # Other predicates default to False
        self.assertFalse(algo.is_ngram())
        self.assertFalse(algo.is_dflash())

    def test_subclass_overrides_create_worker(self):
        name = _unique_name("MY_CUSTOM")

        class CustomDispatch(CustomSpecAlgo):
            def create_worker(self, server_args):
                return "custom-dispatched"

        @SpeculativeAlgorithm.register(name, spec_class=CustomDispatch)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string(name)
        # Custom dispatch bypasses default overlap check
        self.assertEqual(algo.create_worker(MagicMock()), "custom-dispatched")


class TestCrossTypeIdentity(_RegistryIsolated):
    """A plugin algo and a builtin enum value must never compare equal."""

    def test_plugin_not_equal_to_builtin(self):
        name = _unique_name("MY_FOO")

        @SpeculativeAlgorithm.register(name)
        def _factory(server_args):
            return MagicMock

        algo = SpeculativeAlgorithm.from_string(name)
        self.assertNotEqual(algo, SpeculativeAlgorithm.EAGLE)
        self.assertNotEqual(algo, SpeculativeAlgorithm.NONE)
        self.assertIsNot(algo, SpeculativeAlgorithm.EAGLE)


if __name__ == "__main__":
    unittest.main(verbosity=3)
