from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from utils.graph_ops import graph_mean_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def reference_mean(x, batch):
    """Independent per-group reference, including zeros for absent groups."""
    groups = int(batch.max()) + 1 if batch.numel() else 0
    if not groups:
        return x[:0]
    return torch.stack([
        x[batch == group].mean(0) if (batch == group).any() else x.sum(0) * 0
        for group in range(groups)
    ])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("ids", [[0, 0, 1, 1, 1], [3, 0, 3, 2, 0], [0], []])
def test_pool_values_and_gradients(dtype, ids):
    batch = torch.tensor(ids, dtype=torch.long)
    # Slicing creates non-contiguous node features.
    x = torch.randn(len(ids), 8, dtype=dtype)[:, ::2].detach().requires_grad_()
    actual = graph_mean_pool(x, batch)
    expected = reference_mean(x, batch)
    assert actual.dtype == dtype and actual.device == x.device
    torch.testing.assert_close(actual, expected)
    weight = torch.randn_like(actual)
    actual_grad, = torch.autograd.grad((actual * weight).sum(), x)
    expected_grad, = torch.autograd.grad((expected * weight).sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_masked_nodes_still_count_in_mean():
    x = torch.tensor([[6.0, 3.0], [8.0, 4.0], [2.0, 1.0]], requires_grad=True)
    mask = torch.tensor([1.0, 0.0, 0.0])
    pooled = graph_mean_pool(x * mask[:, None], torch.tensor([0, 0, 0]))
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 1.0]]))
    pooled.sum().backward()
    torch.testing.assert_close(x.grad, mask[:, None].expand_as(x) / 3)


def test_pool_gradcheck():
    x = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    batch = torch.tensor([2, 0, 2, 0, 2])
    assert torch.autograd.gradcheck(lambda value: graph_mean_pool(value, batch), (x,))


@pytest.fixture
def single_thread_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def baseline_module(request, monkeypatch):
    baseline, relative_path = request.param
    baseline_root = REPO_ROOT / "baselines" / baseline
    monkeypatch.syspath_prepend(str(baseline_root))
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("pool_test_model", baseline_root / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    # RRD-N's upstream imports use generic package names; do not leak them.
    for name in set(sys.modules) - before:
        if name.split(".")[0] in {"Process", "tools"}:
            del sys.modules[name]


def assert_pooling_equivalent(module, monkeypatch, model, run):
    """Compare full branch outputs and parameter gradients with a loop oracle."""
    def evaluate():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(17)
        outputs = run()
        assert all(torch.isfinite(output).all() for output in outputs)
        sum(output.square().mean() for output in outputs).backward()
        gradients = {name: value.grad.clone() for name, value in model.named_parameters()
                     if value.grad is not None}
        assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
        return [output.detach() for output in outputs], gradients

    actual, actual_gradients = evaluate()
    with monkeypatch.context() as patch:
        patch.setattr(module, "graph_mean_pool", reference_mean)
        expected, expected_gradients = evaluate()
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name in actual_gradients:
        torch.testing.assert_close(actual_gradients[name], expected_gradients[name],
                                   rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("baseline_module", [("BCFN", "model/bigcn.py")], indirect=True)
@pytest.mark.parametrize("training", [False, True])
def test_bcfn_graph_branch(baseline_module, monkeypatch, tmp_path, training):
    module = baseline_module
    data = Data(x=torch.arange(1, 7, dtype=torch.float).reshape(-1, 1),
                edge_index=torch.tensor([[0, 0, 3, 3], [1, 2, 4, 5]]),
                batch=torch.tensor([0, 0, 0, 1, 1, 1]), root_index=torch.tensor([0, 3]))
    model = module.BiGCN(in_channels=2, hidden_size=4, out_channels=4, out_dim=3)
    model.train(training)
    assert_pooling_equivalent(module, monkeypatch, model, lambda: (model(data),))
    checkpoint = tmp_path / "bcfn.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = module.BiGCN(in_channels=2, hidden_size=4, out_channels=4, out_dim=3)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(data), restored(data))


@pytest.mark.parametrize("baseline_module", [("RRD-N", "Model/Bimodel.py")], indirect=True)
@pytest.mark.parametrize("training", [False, True])
def test_rrd_network_and_classifier(baseline_module, monkeypatch, tmp_path, training, single_thread_cpu):
    module = baseline_module
    torch.manual_seed(7)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    edges = torch.tensor([[0, 0, 3, 3], [1, 2, 4, 5]])
    data = Data(x=torch.randn(6, 5000), batch=batch, edge_index=edges,
                TD_edge_index=edges[:, ::2], BU_edge_index_ori=edges.flip(0),
                BU_edge_index=edges.flip(0)[:, ::2], rootindex=torch.tensor([0, 3]),
                mask=torch.tensor([0, 1, 3, 4]), y=torch.tensor([0, 0]))
    net = module.Net(hidden_dim=4, num_gc_layers=3, device="cpu")
    classifier = module.Classfier(12, 4, 2, net.extractor, "cpu")
    model = torch.nn.ModuleDict({"net": net, "classifier": classifier})
    model.train(training)

    def run():
        loss, _ = net(data, data.rootindex, epoch=0, yes=False)
        _, td = net.td_encoder(data.x, edges, batch, data.rootindex)
        _, bu = net.bu_encoder(data.x, edges.flip(0), batch, data.rootindex)
        logits, probabilities = classifier(td, bu, data)
        loss_logits, svdd, loss_probabilities = classifier.loss(td, bu, data)
        embedding = classifier.getembed(td, bu, data)
        assert logits.shape == (2, 2) and embedding.shape == (2, 8)
        torch.testing.assert_close(probabilities.sum(1), torch.ones(2))
        return loss, logits, probabilities, loss_logits, svdd, loss_probabilities, embedding

    assert_pooling_equivalent(module, monkeypatch, model, run)
    checkpoint = tmp_path / "rrd_n.pt"
    torch.save(model.state_dict(), checkpoint)
    restored_net = module.Net(hidden_dim=4, num_gc_layers=3, device="cpu")
    restored = torch.nn.ModuleDict({
        "net": restored_net,
        "classifier": module.Classfier(12, 4, 2, restored_net.extractor, "cpu"),
    })
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    model.eval()
    restored.eval()
    _, td = net.td_encoder(data.x, edges, batch, data.rootindex)
    _, bu = net.bu_encoder(data.x, edges.flip(0), batch, data.rootindex)
    torch.testing.assert_close(classifier(td, bu, data), restored["classifier"](td, bu, data))


@pytest.mark.parametrize("baseline,script", [("BCFN", "train_bcfn.py"), ("RRD-N", "train_rrd_n.py")])
@pytest.mark.parametrize("external_cwd", [False, True])
def test_training_help_imports(baseline, script, external_cwd, tmp_path):
    baseline_root = REPO_ROOT / "baselines" / baseline
    result = subprocess.run([sys.executable, "-B", str(baseline_root / script), "--help"],
                            cwd=tmp_path if external_cwd else baseline_root,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
