"""Extract and score official WavLM speaker-verification variants."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_embedding_quality import norm, leave_one_out_scores, pair_stats, NUM_CLASSES
from sklearn.metrics import roc_auc_score
DATA=ROOT/'data/processed'; AUDIO=DATA/'audio_wav'; OUT=ROOT/'reports/generated'

def score(emb,y):
    emb=norm(emb.astype(np.float32)); ki,sims,top1,top5,cent=leave_one_out_scores(emb,y)
    kd=1-sims.max(1); ui=np.flatnonzero(y==0); ud=1-(emb[ui]@cent.T).max(1)
    auc=float(roc_auc_score(np.r_[np.zeros(len(kd)),np.ones(len(ud))],np.r_[kd,ud]))
    best=(-1,None)
    for t in np.arange(.02,.981,.01):
        kp=top1.copy(); kp[kd>=t]=0
        up=(emb[ui]@cent.T).argmax(1)+1; up[ud>=t]=0
        pred=np.r_[kp,up]; truth=np.r_[y[ki],np.zeros(len(ui),dtype=np.int64)]
        # macro F1 without importing project metric's unknown handling
        from sklearn.metrics import f1_score
        f=float(f1_score(truth,pred,average='macro',zero_division=0))
        if f>best[0]: best=(f,float(t))
    return {'n_files':int(len(y)),'embedding_dim':int(emb.shape[1]),'known_loo_top1':float((top1==y[ki]).mean()),'known_loo_top5':float(np.mean([y[ix] in row for ix,row in zip(ki,top5)])),'ood_auc':auc,'best_macro_f1':{'macro_f1':best[0],'threshold':best[1]},'pair_separation':pair_stats(emb,y)}

def main():
    trf=json.loads((DATA/'train_emb_campp_files.json').read_text()); vaf=json.loads((DATA/'val_campp_vad_files.json').read_text()); files=trf+vaf
    y=np.r_[np.load(DATA/'embeddings_train_campp_labels.npy'),np.load(DATA/'embeddings_val_campp_labels.npy')].astype(np.int64)
    assert len(files)==len(y)
    device='cuda' if torch.cuda.is_available() else 'cpu'; batch=4; maxlen=16000*8
    for name in ['wavlm_base_sv','wavlm_base_plus_sv']:
        path=ROOT/'weights'/name; fe=AutoFeatureExtractor.from_pretrained(path,local_files_only=True); model=AutoModelForAudioXVector.from_pretrained(path,local_files_only=True).to(device).eval(); out=[]
        with torch.inference_mode():
            for i in range(0,len(files),batch):
                waves=[]
                for fn in files[i:i+batch]:
                    x,_=sf.read(AUDIO/fn,dtype='float32'); x=np.asarray(x).reshape(-1)
                    if len(x)>maxlen: start=(len(x)-maxlen)//2; x=x[start:start+maxlen]
                    waves.append(x)
                z=fe(waves,sampling_rate=16000,return_tensors='pt',padding=True)
                z={k:v.to(device) for k,v in z.items()}; out.append(model(**z).embeddings.detach().cpu().numpy())
                if (i//batch)%100==0: print(name,i,'/',len(files),flush=True)
        emb=np.concatenate(out); np.save(DATA/f'embeddings_{name}.npy',emb); result=score(emb,y); (OUT/f'embedding_quality_{name}.json').write_text(json.dumps(result,indent=2)); print(json.dumps({name:result},indent=2))
if __name__=='__main__': main()
