from cogeneration.scripts.train import Experiment


class TestExperiment:
    def test_constructor(self, mock_cfg):
        exp = Experiment(cfg=mock_cfg)
        assert exp is not None
