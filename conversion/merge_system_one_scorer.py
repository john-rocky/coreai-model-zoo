#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "torch==2.9.0", "transformers==5.17.0", "peft==0.21.0",
#   "safetensors==0.8.0", "huggingface_hub==1.32.0", "numpy==2.3.5",
#   "tokenizers==0.23.2", "accelerate==1.15.0",
# ]
# ///
"""Merge pngwn/system-one-qwen3.5-4b-scorer (cc-by-nc-4.0) on CPU in fp32.

The pinned adapter supplies 200 LoRA A/B pairs across ten projection families
and score.weight [1,2560]. The base is Qwen/Qwen3.5-4B-Base at the revision
below. in_proj_a and in_proj_b receive no LoRA. Import the pinned author's
system_one.py unchanged, build load_model(base, lora=False, dtype=float32),
and restore the adapter's score through PEFT. A separate merged copy must
agree with the adapter on three author-encoded sequences within 1e-3 before
serialization. Save one model.safetensors: model.* text tower in bfloat16,
score.weight in float32; the flat text config has tie_word_embeddings=false.
The parity check precedes BF16 storage rounding; bundle readout checks include
storage/export rounding. No visual tower or vocabulary output head is saved.

CLI: uv run --script merge_system_one_scorer.py --base-snapshot BASE \
     --adapter-snapshot ADAPTER --out MERGED
"""
from __future__ import annotations
import argparse, copy, gc, hashlib, importlib.util, json, math, os, time
from pathlib import Path
LICENSE='cc-by-nc-4.0'
ADAPTER_ID='pngwn/system-one-qwen3.5-4b-scorer'
ADAPTER_REV='e6464dce15f013c2ef641593a85cc6afcdaea928'
BASE_ID='Qwen/Qwen3.5-4B-Base'
BASE_REV='1001bb4d826a52d1f399e183466143f4da7b741b'
PINNED_FILES={
 'adapter':{
  'adapter_config.json':'62607c5d76f2b1fdd81059361c8a2da5e99f3a56f35420893657f88c1ff949ce',
  'adapter_model.safetensors':'323174d808153e97cc24033ffd067f284bff416632eebc6d6f142e551aa91c6a',
  'system_one.py':'cd9865bc82e1b49972955986e74856f10f0a66d4b9d057114958d1b4625906ff'},
 'base':{
  'config.json':'ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670',
  'model.safetensors-00001-of-00002.safetensors':'df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712',
  'model.safetensors-00002-of-00002.safetensors':'590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef'}}
MERGE_CASES=[
 ('A seed sprouted roots and two green leaves.','Which topic fits?','plants'),
 ('A broken door lock prevents entry to the room.','Which category fits?','lock'),
 ('Heavy rain left puddles along the path.','Which condition is described?','rain')]
TARGETS={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj'}
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def verify_inputs(base,adapter):
 records=[]
 for kind,directory in [('base',base),('adapter',adapter)]:
  for name,expected in PINNED_FILES[kind].items():
   path=directory/name;actual=sha(path);assert actual==expected,f'Pinned source mismatch: {path}'
   records.append({'kind':kind,'name':name,'bytes':path.stat().st_size,'sha256':actual,'match':True})
 return records
def load_author(adapter):
 path=Path(adapter)/'system_one.py';assert sha(path)==PINNED_FILES['adapter']['system_one.py']
 spec=importlib.util.spec_from_file_location('pinned_system_one_author',path)
 author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author);return author
def merge(base,adapter,out,report):
 started=time.monotonic();base=Path(base).resolve();adapter=Path(adapter).resolve();out=Path(out).resolve()
 assert out not in (base,adapter),'Do not overwrite a source snapshot'
 receipt={'schema':'coreai-scalar-merge/1','status':'RUNNING','license':LICENSE,'adapter':{'repo':ADAPTER_ID,'revision':ADAPTER_REV},'base':{'repo':BASE_ID,'revision':BASE_REV},'source_files':verify_inputs(base,adapter),'device':'cpu','compute_dtype':'float32','torch_threads':8,'tolerance':1e-3}
 dump(report,receipt)
 import torch
 from peft import PeftModel
 from safetensors.torch import load_file,save_file
 from transformers import AutoTokenizer
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(0)
 author=load_author(adapter)
 tok,model=author.load_model(str(base),lora=False,dtype=torch.float32)
 oracle=PeftModel.from_pretrained(model,str(adapter)).eval();del model
 assert next(oracle.parameters()).device.type=='cpu'
 assert {p.dtype for p in oracle.parameters()if p.is_floating_point()}=={torch.float32}
 tok_adapter=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
 assert tok.pad_token_id==tok.eos_token_id==tok_adapter.pad_token_id==tok_adapter.eos_token_id
 state=load_file(str(adapter/'adapter_model.safetensors'),device='cpu')
 a=[k for k in state if '.lora_A.' in k];b=[k for k in state if '.lora_B.' in k]
 assert len(state)==401 and len(a)==len(b)==200
 families={k.split('.lora_A.')[0].rsplit('.',1)[-1]for k in a};assert families==TARGETS
 score=oracle.get_base_model().score;head=score.modules_to_save['default'].weight
 assert head.shape==(1,2560) and torch.equal(head.detach(),state['base_model.model.score.weight'].float())
 del state
 merged=copy.deepcopy(oracle).merge_and_unload(safe_merge=True).eval()
 sequences=[author.encode(tok,*case,384)for case in MERGE_CASES]
 assert sequences==[author.encode(tok_adapter,*case,384)for case in MERGE_CASES]
 assert all(tok.pad_token_id not in ids for ids in sequences)
 ids,mask=author.pad_batch(sequences,tok.pad_token_id,384)
 captured=[]
 hook=score.register_forward_hook(lambda mod,inputs,output:captured.append(output.detach().float().cpu()))
 try:
  with torch.no_grad():adapter_scores=oracle(input_ids=ids,attention_mask=mask).logits.squeeze(-1).float().tolist()
 finally:hook.remove()
 assert len(captured)==1
 pooled=[float(captured[0][i,len(seq)-1,0])for i,seq in enumerate(sequences)]
 assert pooled==adapter_scores
 with torch.no_grad():merged_scores=merged(input_ids=ids,attention_mask=mask).logits.squeeze(-1).float().tolist()
 errors=[abs(x-y)for x,y in zip(adapter_scores,merged_scores)]
 receipt.update(adapter_tensors={'total':401,'lora_a':200,'lora_b':200,'lora_pairs':200,'projection_families':sorted(families),'untargeted':['in_proj_a','in_proj_b']},score_head={'shape':[1,2560],'bias':False,'source_dtype':'bfloat16','restored_exactly':True},sequences=[{'state':case[0],'question':case[1],'option':case[2],'ids':seq,'slot':len(seq)-1,'adapter_fp32':x,'merged_fp32':y,'absolute_error':e,'head_last_non_pad':p}for case,seq,x,y,e,p in zip(MERGE_CASES,sequences,adapter_scores,merged_scores,errors,pooled)],max_absolute_error=max(errors),all_finite=all(math.isfinite(x)for x in adapter_scores+merged_scores),pooling={'verified':True,'right_padding':True,'pad_token_id':tok.pad_token_id,'eos_token_id':tok.eos_token_id},author_script_sha256=sha(adapter/'system_one.py'))
 assert receipt['all_finite'] and max(errors)<=1e-3,receipt
 del oracle,captured;gc.collect()
 weights={}
 for key,value in merged.state_dict().items():
  assert (key.startswith('model.') or key=='score.weight') and not any(x in key for x in ('language_model','lora_','modules_to_save'))
  weights[key]=value.detach().cpu().to(torch.float32 if key=='score.weight'else torch.bfloat16).contiguous()
 assert weights['model.embed_tokens.weight'].shape==(248320,2560)
 assert all(torch.isfinite(v).all().item()for v in weights.values())
 out.mkdir(parents=True,exist_ok=True);tmp=out/'model.safetensors.tmp'
 save_file(weights,str(tmp),metadata={'format':'pt','license':LICENSE,'base_revision':BASE_REV,'adapter_revision':ADAPTER_REV,'tower_dtype':'bfloat16','score_dtype':'float32'});tmp.replace(out/'model.safetensors')
 config=copy.deepcopy(json.loads((base/'config.json').read_text())['text_config'])
 config.update(model_type='qwen3_5_text',architectures=['Qwen3_5TextForSequenceClassification'],tie_word_embeddings=False,num_labels=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,dtype='bfloat16',license=LICENSE,merged_from={'base':f'{BASE_ID}@{BASE_REV}','adapter':f'{ADAPTER_ID}@{ADAPTER_REV}','compute_dtype':'float32','tower_storage_dtype':'bfloat16','score_storage_dtype':'float32'})
 dump(out/'config.json',config)
 receipt.update(status='PASS',merged_file=str(out/'model.safetensors'),sha256=sha(out/'model.safetensors'),size_bytes=(out/'model.safetensors').stat().st_size,tensor_count=len(weights),tower_storage_dtype='bfloat16',score_storage_dtype='float32',storage_note='FP32 merge parity is checked before BF16 tower storage; bundle readout includes storage/export rounding.',config_sha256=sha(out/'config.json'),wall_seconds=time.monotonic()-started)
 dump(report,receipt);print(json.dumps({'status':'PASS','merged_file':receipt['merged_file'],'max_absolute_error':max(errors),'sha256':receipt['sha256'],'wall_seconds':receipt['wall_seconds']}),flush=True)
 return receipt
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-snapshot',required=True);p.add_argument('--adapter-snapshot',required=True);p.add_argument('--out',required=True);p.add_argument('--report');a=p.parse_args()
 report=Path(a.report)if a.report else Path(a.out)/'merge.json'
 try:merge(a.base_snapshot,a.adapter_snapshot,a.out,report)
 except Exception as exc:
  old=json.loads(report.read_text())if report.exists()else{};old.update(status='FAIL',error=f'{type(exc).__name__}: {exc}');dump(report,old);raise
if __name__=='__main__':main()
