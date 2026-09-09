from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.peft_fullft_reference_v3 import model as module


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(4,4)
        self.output_layer = nn.Linear(4,6)
        self.residual_layer = nn.Linear(4,6)
        self.calls = 0

    def forward(self,x):
        self.calls += 1
        return self.output_layer(self.hidden_layer(x).relu()) + self.residual_layer(x)


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(dropout_rate=0.)
        self.chronos_config = SimpleNamespace(output_patch_size=2,use_arcsinh=True)
        self.num_quantiles = 3
        self.device = torch.device("cpu")
        self.input_patch_embedding = nn.Linear(4,4)
        self.encoder = nn.Linear(4,4)
        self.output_patch_embedding = TinyHead()

    def encode(self,context,group_ids,num_output_patches):
        hidden = self.encoder(self.input_patch_embedding(context))[:,None,:].repeat(1,num_output_patches,1)
        shape = (len(context),1)
        return SimpleNamespace(last_hidden_state=hidden),(torch.zeros(shape),torch.ones(shape)),None,None


@pytest.fixture
def base(monkeypatch):
    torch.manual_seed(17)
    base = TinyBase()
    head = sum(p.numel() for p in base.output_patch_embedding.parameters())
    total = sum(p.numel() for p in base.parameters())
    monkeypatch.setitem(module.EXPECTED_COUNTS,"HEAD_ONLY",head)
    monkeypatch.setitem(module.EXPECTED_COUNTS,"FULL_FT",total)
    monkeypatch.setitem(module.native.EXPECTED_TRAINABLE,"H_FULL",head)
    return base


@pytest.mark.parametrize("arm",["F0","HEAD_ONLY","FULL_FT"])
def test_scope_and_stepzero_identity_call_native_head_once(base,arm):
    initial = deepcopy(base)
    context,groups = torch.randn(3,4),torch.arange(3)
    hidden,_,_,_ = initial.encode(context,groups,2)
    expected = module.native.patch_to_quantiles(initial.output_patch_embedding(hidden.last_hidden_state),3,2)
    adapted = module.construct(base,arm,20000)
    actual,_,_,_ = module.forward(adapted,context,groups,horizon=4)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert base.output_patch_embedding.calls == 1
    scope = module.audit_scope(adapted,arm)
    assert scope["trainable"] == module.EXPECTED_COUNTS[arm]
    assert (scope["frozen_parameters"] == 0) == (arm == "FULL_FT")


@pytest.mark.parametrize("arm",["HEAD_ONLY","FULL_FT"])
def test_optimizer_updates_intended_native_components_only(base,arm):
    adapted = module.construct(base,arm,20000)
    encoder_before = module.parameter_digest(adapted,prefix="base.encoder.")
    head_before = module.parameter_digest(adapted,prefix="base.output_patch_embedding.")
    optimizer = torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad],lr=.1)
    norm,_,_,_ = module.forward(adapted,torch.randn(3,4),torch.arange(3),4)
    norm.square().mean().backward()
    optimizer.step()
    assert module.parameter_digest(adapted,prefix="base.output_patch_embedding.") != head_before
    assert (module.parameter_digest(adapted,prefix="base.encoder.") != encoder_before) == (arm == "FULL_FT")
    if arm == "HEAD_ONLY":
        assert all(p.grad is None for p in base.encoder.parameters())


def test_cpu_best_snapshot_is_independent_and_disk_restore_exact(base,tmp_path):
    adapted = module.construct(base,"FULL_FT",20000)
    before = module.parameter_digest(adapted)
    state = module.snapshot_trainable(adapted)
    with torch.no_grad():
        for p in adapted.parameters():
            p.add_(1)
    assert module.parameter_digest(adapted) != before
    path = tmp_path/"best_trainable.pt"
    module.write_checkpoint(state,path)
    module.restore_trainable(adapted,torch.load(path,map_location="cpu",weights_only=True))
    assert module.parameter_digest(adapted) == before
    with pytest.raises(FileExistsError):
        module.write_checkpoint(state,path)


@pytest.mark.parametrize("damage",["missing","shape","dtype","nan"])
def test_restore_rejects_invalid_parameter_contract(base,damage):
    adapted = module.construct(base,"FULL_FT",20000)
    state = module.snapshot_trainable(adapted)
    key = next(iter(state))
    if damage == "missing":
        del state[key]
    elif damage == "shape":
        state[key] = state[key].reshape(-1)
    elif damage == "dtype":
        state[key] = state[key].double()
    else:
        state[key].fill_(float("nan"))
    with pytest.raises((AssertionError,FloatingPointError)):
        module.restore_trainable(adapted,state)


@pytest.mark.parametrize("damage",["selection","holdout"])
def test_forecast_blocks_invalid_selection_or_holdout_before_panel_load(tmp_path,monkeypatch,damage):
    from experiments.peft_fullft_reference_v3 import forecast,train
    monkeypatch.setattr(forecast,"ROOT",tmp_path)
    monkeypatch.setattr(train,"ROOT",tmp_path)
    root = tmp_path/"runs"/forecast.STUDY
    root.mkdir(parents=True)
    contract_path = root/"study_contract.json"
    contract_path.write_text("{}",encoding="utf-8")
    holdout = root/"holdout.npz"
    holdout.write_bytes(b"must never be parsed")
    selected = {"completed":True,"global_choices_frozen":damage != "selection",
                "contract_sha256":forecast.digest(contract_path),"selected":[]}
    (root/"selection.json").write_text(json.dumps(selected),encoding="utf-8")
    contract = {"datasets":{"bike":{"holdout_data_path":str(holdout),"holdout_data_sha256":"changed"}}}
    monkeypatch.setattr(forecast,"read_contract",lambda *a,**kw:contract)
    def forbidden_panel(*a,**kw):
        pytest.fail("Holdout Panel opened before its authorization/hash gate")
    monkeypatch.setattr(forecast.shared,"Panel",forbidden_panel)
    args = SimpleNamespace(contract=contract_path,dataset="bike",arm="F0",fit_dir=None,output=root/"forecast")
    with pytest.raises(AssertionError,match="Global model selection|Holdout archive"):
        forecast.execute(args)


@pytest.mark.parametrize("arm",["HEAD_ONLY","LORA"])
def test_adapted_forecast_requires_verified_frozen_backbone(tmp_path,monkeypatch,arm):
    from experiments.peft_fullft_reference_v3 import forecast,train
    monkeypatch.setattr(forecast,"ROOT",tmp_path)
    monkeypatch.setattr(train,"ROOT",tmp_path)
    root = tmp_path/"runs"/forecast.STUDY
    fit_dir = root/"fit"
    fit_dir.mkdir(parents=True)
    contract_path = root/"study_contract.json"
    contract_path.write_text("{}",encoding="utf-8")
    checkpoint = fit_dir/"best_trainable.pt"
    checkpoint.write_bytes(b"fixture")
    fit = {"completed":True,"smoke":False,"stage":"fit","contract_sha256":forecast.digest(contract_path),
           "holdout_file_opened":False,"dataset":"bike","arm":arm,"seed":20000,"steps_completed":200,"lr":1e-5,
           "audits":{name:True for name in ("zero_update_identity","trainable_map_verified","optimizer_exact_parameter_set",
                                            "finite_nonzero_gradient_verified","checkpoint_reload_verified")}}
    fit["audits"].update(frozen_parameters_verified=False,frozen_check_applicable=True)
    result = fit_dir/"result.json"
    result.write_text(json.dumps(fit),encoding="utf-8")
    selected = {"completed":True,"global_choices_frozen":True,"contract_sha256":forecast.digest(contract_path),
                "selected":[{"dataset":"bike","arm":arm,"seed":20000,"fit_dir":str(fit_dir),
                             "fit_result_sha256":forecast.digest(result),"checkpoint_sha256":forecast.digest(checkpoint)}]}
    (root/"selection.json").write_text(json.dumps(selected),encoding="utf-8")
    contract = {"datasets":{"bike":{}},"settings":{"steps":200,"seeds":[20000],"lr_grids":{arm:[1e-5]}}}
    with pytest.raises(AssertionError,match="frozen-backbone"):
        forecast.authorize_selection(contract_path,contract,"bike",arm,fit_dir)


@pytest.mark.parametrize("arm",["HEAD_ONLY","FULL_FT"])
def test_mapped_best_capture_updates_same_file_storage_only_when_requested(base,tmp_path,arm):
    adapted = module.construct(base,arm,20000)
    path = tmp_path/"best_state.bin"
    mapped = module.MappedBestState(adapted,path)
    try:
        mapped.capture(adapted)
        parameters = {name:p for name,p in adapted.named_parameters() if p.requires_grad}
        expected_bytes = sum(p.numel()*p.element_size() for p in parameters.values())
        assert mapped.path == path
        assert mapped.nbytes == path.stat().st_size == expected_bytes
        assert set(mapped.state) == set(parameters)
        offset = 0
        pointers = {}
        initial = {}
        for name,parameter in parameters.items():
            value = mapped.state[name]
            assert value.device.type == "cpu" and value.dtype == torch.float32
            assert value.shape == parameter.shape
            assert value.data_ptr() == mapped._flat.ctypes.data + offset
            pointers[name] = value.data_ptr()
            initial[name] = value.clone()
            offset += parameter.numel()*parameter.element_size()
        with torch.no_grad():
            for parameter in parameters.values():
                parameter.add_(1)
        for name in parameters:
            torch.testing.assert_close(mapped.state[name],initial[name],rtol=0,atol=0)
        mapped.capture(adapted)
        assert path.stat().st_size == expected_bytes
        for name,parameter in parameters.items():
            assert mapped.state[name].data_ptr() == pointers[name]
            torch.testing.assert_close(mapped.state[name],parameter,rtol=0,atol=0)
    finally:
        mapped.close()


def test_mapped_best_writes_once_and_mmap_checkpoint_restores_exactly(base,tmp_path):
    adapted = module.construct(base,"FULL_FT",20000)
    before = module.parameter_digest(adapted)
    mapped = module.MappedBestState(adapted,tmp_path/"best_state.bin")
    checkpoint = tmp_path/"best_trainable.pt"
    try:
        mapped.capture(adapted)
        module.write_checkpoint(mapped.state,checkpoint)
        checkpoint_bytes = checkpoint.read_bytes()
        with pytest.raises(FileExistsError):
            module.write_checkpoint(mapped.state,checkpoint)
        assert checkpoint.read_bytes() == checkpoint_bytes
    finally:
        mapped.close()
    with torch.no_grad():
        for parameter in adapted.parameters():
            parameter.add_(1)
    reloaded = torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True)
    module.restore_trainable(adapted,reloaded)
    assert module.parameter_digest(adapted) == before


def test_mapped_best_close_flushes_retains_file_and_releases_handle(base,tmp_path):
    adapted = module.construct(base,"FULL_FT",20000)
    path = tmp_path/"best_state.bin"
    mapped = module.MappedBestState(adapted,path)
    mapped.capture(adapted)
    expected = b"".join(value.numpy().tobytes() for value in mapped.state.values())
    mapped.close()
    assert mapped.state == {}
    assert path.read_bytes() == expected
    mapped.close()
    moved = tmp_path/"closed_state.bin"
    path.rename(moved)
    moved.rename(path)
    assert path.read_bytes() == expected


@pytest.mark.parametrize("invalid",["float64","existing_file"])
def test_mapped_best_rejects_invalid_construction_without_truncation(base,tmp_path,invalid):
    adapted = module.construct(base,"FULL_FT",20000)
    path = tmp_path/"best_state.bin"
    if invalid == "float64":
        adapted.double()
        with pytest.raises((TypeError,ValueError,AssertionError)):
            module.MappedBestState(adapted,path)
        assert not path.exists()
    else:
        original = b"preserve previous best mapping"
        path.write_bytes(original)
        with pytest.raises(FileExistsError):
            module.MappedBestState(adapted,path)
        assert path.read_bytes() == original


@pytest.mark.parametrize("available_gib",[5.999,6.0])
def test_commit_headroom_rejects_below_six_and_accepts_exact_boundary(tmp_path,monkeypatch,available_gib):
    from experiments.peft_adaptation_scope_v1 import guard
    from experiments.peft_fullft_reference_v3 import train
    available_bytes = int(available_gib*1024**3)
    monkeypatch.setattr(guard,"_available_commit",lambda:available_bytes)
    monkeypatch.setattr(train,"_available_commit",lambda:available_bytes,raising=False)
    monkeypatch.setattr(train,"memory_phase",lambda *a,**kw:None)
    monkeypatch.setattr(train.shared,"log",lambda *a,**kw:None)
    if available_gib < 6:
        with pytest.raises(RuntimeError):
            train.check_commit_headroom(tmp_path,"before_checkpoint")
    else:
        train.check_commit_headroom(tmp_path,"before_checkpoint")
