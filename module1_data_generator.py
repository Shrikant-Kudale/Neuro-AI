"""
MODULE 1 — Synthetic EEG Data Generator
Generates 900 synthetic EEG subjects (300 per class).
Optimized: direct oscillator synthesis, no scipy filtering loops.
"""
import numpy as np
import pandas as pd
import os

FS=256; DURATION=30; N_CHANNELS=19; N_SUBJECTS=300; RANDOM_SEED=42

def generate_dataset(save_dir='outputs/data'):
    os.makedirs(save_dir, exist_ok=True)
    rng=np.random.default_rng(RANDOM_SEED)
    n_samples=FS*DURATION
    t=np.arange(n_samples)/FS
    class_names=['Normal','Early Disruption','High Risk']
    mi_p={(0):(0.18,0.04),(1):(0.11,0.03),(2):(0.05,0.02)}
    bp={(0):dict(etvs=(1.20,0.18),gai=(0.04,0.010),sscs=(0.22,0.04)),
        (1):dict(etvs=(1.65,0.22),gai=(0.09,0.020),sscs=(0.34,0.05)),
        (2):dict(etvs=(2.30,0.31),gai=(0.17,0.030),sscs=(0.47,0.07))}
    all_eeg=np.zeros((900,N_CHANNELS,n_samples),dtype=np.float32)
    all_lbls=np.zeros(900,dtype=np.int64)
    all_beh=np.zeros((900,3),dtype=np.float32)
    records=[]; sid=0
    for cls in range(3):
        print(f"  [{cls+1}/3] Generating {N_SUBJECTS} {class_names[cls]} subjects...")
        mm,ms=mi_p[cls]
        for i in range(N_SUBJECTS):
            eeg=np.zeros((N_CHANNELS,n_samples),dtype=np.float32)
            for ch in range(N_CHANNELS):
                d=rng.normal(1.3,0.2)*np.sin(2*np.pi*rng.uniform(1,3)*t+rng.uniform(0,6.28))
                th=rng.normal(0.9,0.15)*np.sin(2*np.pi*rng.uniform(5,7)*t+rng.uniform(0,6.28))
                a=rng.normal(0.7,0.12)*np.sin(2*np.pi*rng.uniform(9,12)*t+rng.uniform(0,6.28))
                b=rng.normal(0.5,0.10)*np.sin(2*np.pi*rng.uniform(15,25)*t+rng.uniform(0,6.28))
                g=rng.normal(0.25,0.06)*np.sin(2*np.pi*rng.uniform(35,65)*t+rng.uniform(0,6.28))
                mi=max(0.01,rng.normal(mm,ms))
                if cls>0 and ch<7: mi*=rng.uniform(0.3,0.7)
                tp=2*np.pi*rng.uniform(5,7)*t+rng.uniform(0,6.28)
                g=g*(1.0+mi*np.cos(tp))
                sig=d+th+a+b+g+rng.normal(0,0.05,n_samples)
                sig=sig/(np.std(sig)+1e-8)*rng.uniform(20,55)
                eeg[ch]=sig.astype(np.float32)
            all_eeg[sid]=eeg; all_lbls[sid]=cls
            b2=bp[cls]
            etvs=float(np.clip(rng.normal(*b2['etvs']),0.5,4.0))
            gai=float(np.clip(rng.normal(*b2['gai']),0.0,0.5))
            sscs=float(np.clip(rng.normal(*b2['sscs']),0.0,1.0))
            all_beh[sid]=[etvs,gai,sscs]
            records.append({'subject_id':sid,'class':cls,'class_name':class_names[cls],
                            'etvs':round(etvs,4),'gai':round(gai,4),'sscs':round(sscs,4)})
            sid+=1
    np.save(os.path.join(save_dir,'eeg_data.npy'),all_eeg)
    np.save(os.path.join(save_dir,'labels.npy'),all_lbls)
    np.save(os.path.join(save_dir,'behavioral.npy'),all_beh)
    pd.DataFrame(records).to_csv(os.path.join(save_dir,'behavioral.csv'),index=False)
    print(f"\n  EEG shape: {all_eeg.shape} | Labels: {all_lbls.shape} | Saved to {save_dir}/")
    return all_eeg, all_lbls, pd.DataFrame(records)

if __name__=='__main__':
    print("="*55+"\nMODULE 1 — Synthetic EEG Data Generator\n"+"="*55)
    generate_dataset()
    print("\nModule 1 complete.")
