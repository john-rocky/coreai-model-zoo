"""Load the locally merged text tower and scalar score head for cc-by-nc-4.0 scorer."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
from safetensors import safe_open
from coreai_models.models.macos.qwen3_5 import Qwen3_5StatefulForCausalLM, qwen3_5_config_from_hf
from coreai_models.models.base import _is_layer_key_beyond

HF_ID = 'pngwn/system-one-qwen3.5-4b-scorer'
REVISION = 'e6464dce15f013c2ef641593a85cc6afcdaea928'
BASE = 'Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b'


def load_text_config(snapshot, max_context_length=4096, num_layers=None):
    text = json.loads((Path(snapshot) / 'config.json').read_text())
    assert text['model_type'] == 'qwen3_5_text'
    assert text['vocab_size'] == 248320 and text['hidden_size'] == 2560
    assert text['tie_word_embeddings'] is False
    return qwen3_5_config_from_hf(SimpleNamespace(**text), max_context_length, num_layers)


def load_scorer_model(snapshot, max_context_length=4096, target_dtype=torch.float16, num_layers=None, record_path=None):
    snapshot = Path(snapshot).resolve()
    cfg = load_text_config(snapshot, max_context_length, num_layers)
    model = Qwen3_5StatefulForCausalLM(cfg, model_device='meta')
    model.to(dtype=target_dtype)
    sd, shapes, dtypes, head = {}, {}, {}, None
    prefix = 'model.'
    files = sorted(snapshot.glob('*.safetensors'))
    assert len(files) == 1 and files[0].name == 'model.safetensors'
    for path in files:
        with safe_open(path, framework='pt', device='cpu') as f:
            for key in f.keys():
                shapes[key] = list(f.get_slice(key).get_shape())
                dtypes[key] = f.get_slice(key).get_dtype()
                if key == 'score.weight':
                    head = f.get_tensor(key).to(target_dtype)
                if not key.startswith(prefix):
                    continue
                local = 'model.' + key[len(prefix):]
                if num_layers is not None and _is_layer_key_beyond(local, num_layers):
                    continue
                sd[local] = f.get_tensor(key).to(target_dtype)
    missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
    assert missing == ['lm_head.weight'] and not unexpected, (missing, unexpected)
    assert shapes['model.embed_tokens.weight'] == [248320, 2560]
    assert head is not None and head.shape == (1, 2560)
    # Replace before the original loader's final meta check, as in the sibling.
    model.lm_head = torch.nn.Linear(2560, 1, bias=False, dtype=target_dtype)
    model.lm_head.load_state_dict({'weight': head})
    assert torch.equal(model.lm_head.weight, head)
    assert model.lm_head.bias is None
    model.model.reset_buffers()
    assert not [n for n,p in model.named_parameters() if p.is_meta]
    assert all(torch.isfinite(p).all().item() for p in model.parameters())
    rec = {'hf_id':HF_ID, 'resolved_revision':REVISION, 'base':BASE,
           'snapshot':str(snapshot), 'hf_state_dict_prefix':prefix,
           'tensors':len(shapes), 'shapes':shapes, 'stored_dtypes':dtypes,
           'missing_before_head_replacement':missing, 'unexpected':unexpected,
           'temperature':1.75, 'embedding_rows':248320, 'logits_width':1,
           'dtype':str(target_dtype), 'head_bias':False, 'meta_parameters':[],
           'license':'cc-by-nc-4.0', 'finite_all_parameters':True}
    if record_path:
        Path(record_path).write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rec.items() if k not in ('shapes','stored_dtypes')},indent=2),flush=True)
    return model
