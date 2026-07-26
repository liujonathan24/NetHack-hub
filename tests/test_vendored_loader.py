import pytest


def test_external_harness_ids_resolve():
    from verifiers.v1.loaders import import_harness
    with pytest.raises(ModuleNotFoundError) as e:
        import_harness("definitely-not-installed-harness")
    # The patched loader reports BOTH import candidates.
    assert "verifiers.v1.harnesses." in str(e.value)
    assert "definitely_not_installed_harness" in str(e.value)
