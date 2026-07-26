"""Stock verifiers 0.2.1 already resolves external (non-bundled) plugins.

Task 8 vendored PR #1985's loader patch into `.venv-cli-eval`'s site-packages on
the assumption that stock 0.2.x could only import *bundled* harnesses. It cannot
— stock `_import_plugin` (`verifiers/v1/loaders.py:32-55`) probes the namespaced
candidate with `importlib.util.find_spec` and falls back to the top-level module
name, which is exactly what an external harness/taskset plugin needs. The patch
has been reverted (a patched site-packages is a reproducibility liability for a
benchmark); PR #1985's residual delta is error-reporting quality only.

This test pins the behavior Tasks 9/10 depend on: a plugin installed as an
ordinary top-level package resolves by id, and the built-in ones still resolve
under their namespace.
"""

import sys
import textwrap

import pytest


@pytest.fixture
def external_harness(tmp_path, monkeypatch):
    """An external harness plugin: a top-level module exporting one Harness."""
    (tmp_path / "vf_probe_harness.py").write_text(
        textwrap.dedent(
            """
            from verifiers.v1.harness import Harness

            class ProbeHarness(Harness):
                SUPPORTS_MCP = True

                async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls):
                    raise NotImplementedError

            __all__ = ["ProbeHarness"]
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "vf_probe_harness"
    sys.modules.pop("vf_probe_harness", None)


def test_external_plugin_resolves_without_the_vendored_patch(external_harness):
    from verifiers.v1.loaders import harness_class

    cls = harness_class(external_harness)
    assert cls.__name__ == "ProbeHarness"
    assert cls.__module__ == external_harness


def test_builtin_plugin_still_resolves_under_its_namespace():
    from verifiers.v1.loaders import harness_class

    cls = harness_class("claude_code")
    assert cls.__module__.startswith("verifiers.v1.harnesses.claude_code")
    # The MCP-capable arm this experiment replaced Codex with.
    assert cls.SUPPORTS_MCP is True


def test_missing_plugin_names_the_module_it_tried():
    from verifiers.v1.loaders import import_harness

    with pytest.raises(ModuleNotFoundError) as e:
        import_harness("definitely-not-installed-harness")
    # Hyphens normalize to underscores, and the message names both the module it
    # tried and the built-in namespace it searched.
    assert "definitely_not_installed_harness" in str(e.value)
    assert "verifiers.v1.harnesses" in str(e.value)


def test_site_packages_loader_is_stock():
    """A patched site-packages would silently change plugin resolution for every
    arm; the benchmark must run on the released wheel."""
    import hashlib
    import base64
    import csv
    import pathlib

    import verifiers

    site = pathlib.Path(verifiers.__file__).resolve().parent.parent
    records = list(site.glob("verifiers-*.dist-info/RECORD"))
    if not records:
        pytest.skip("verifiers not installed from a wheel (editable/source install)")
    modified = []
    for row in csv.reader(records[0].open()):
        if len(row) < 3 or not row[1]:
            continue
        path = site / row[0]
        if not path.exists():
            continue
        algo, _, want = row[1].partition("=")
        got = base64.urlsafe_b64encode(
            hashlib.new(algo, path.read_bytes()).digest()
        ).rstrip(b"=").decode()
        if got != want:
            modified.append(row[0])
    assert modified == []
