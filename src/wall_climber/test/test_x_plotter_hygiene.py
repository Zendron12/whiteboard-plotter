from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


def test_x_plotter_modules_do_not_import_vector_pipeline() -> None:
    x_plotter_dir = PACKAGE_ROOT / 'wall_climber' / 'x_plotter'

    for module_path in x_plotter_dir.glob('*.py'):
        tree = ast.parse(module_path.read_text(encoding='utf-8'), filename=str(module_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = {alias.name for alias in node.names}
                assert 'wall_climber.vector_pipeline' not in imported
                assert 'vector_pipeline' not in imported
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = {alias.name for alias in node.names}
                assert module != 'wall_climber.vector_pipeline'
                assert not (module == 'wall_climber' and 'vector_pipeline' in names)
                assert not (node.level > 0 and 'vector_pipeline' in names)


def test_package_docs_live_under_wall_climber_package() -> None:
    docs_dir = PACKAGE_ROOT / 'docs'

    for filename in [
        'README.md',
        'canonical-first-ingestion.md',
        'legacy-compatibility.md',
        'primitive-transport.md',
        'x_plotter_foundation.md',
    ]:
        assert (docs_dir / filename).is_file()


def test_root_docs_are_navigation_not_setup_install_source() -> None:
    assert (REPO_ROOT / 'docs' / 'README.md').is_file()

    setup_source = (PACKAGE_ROOT / 'setup.py').read_text(encoding='utf-8')
    assert "package_files('docs'" in setup_source
    assert "'..', '..', 'docs'" not in setup_source
    assert '../../docs' not in setup_source


def test_tests_are_not_runtime_python_package() -> None:
    assert (PACKAGE_ROOT / 'test').is_dir()
    assert not (PACKAGE_ROOT / 'test' / '__init__.py').exists()

    setup_source = (PACKAGE_ROOT / 'setup.py').read_text(encoding='utf-8')
    assert "find_packages(exclude=['test', 'test.*'])" in setup_source
