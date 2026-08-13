import src.experiment as experiment_api


def test_public_api_exposes_the_framework_without_the_mutable_experiment():
    expected = {
        "RunManifest",
        "CampaignId",
        "ExperimentPlan",
        "ExperimentRunner",
        "ExperimentResult",
        "Stage",
        "load_manifest",
        "plan_experiment",
    }
    assert expected.issubset(set(experiment_api.__all__))
    assert not hasattr(experiment_api, "Experiment")
