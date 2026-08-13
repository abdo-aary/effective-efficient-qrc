from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_current_paper_has_one_canonical_project_and_no_latex_hierarchy():
    assert not (ROOT / "docs/latex").exists()
    assert (ROOT / "docs/paper/main.tex").is_file()
    assert (ROOT / "docs/paper/main.pdf").is_file()
    assert (ROOT / "docs/paper/sections/experiments.tex").is_file()


def test_architecture_and_executable_contract_are_separated():
    assert (ROOT / "docs/architecture/experiment-framework.md").is_file()
    assert (ROOT / "docs/architecture/implementation-contract.md").is_file()
    assert (ROOT / "experiments/empirical_evaluation/experiment_contract.yaml").is_file()

