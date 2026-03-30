"""DOGE SHORT - TP=4% SL=3% adapted for high volatility (6.6% avg daily range)"""
import sys
from pathlib import Path
import pandas as pd, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json, joblib, logging
from sklearn.preprocessing import RobustScaler

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
from direction_prediction_model import CNNDirectionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / 'data' / 'cache'
MODEL_DIR = BASE_DIR / 'models_short'
MODEL_DIR.mkdir(exist_ok=True)

SHORT_TP = 0.04
SHORT_SL = 0.03
SEQ = 30

def create_labels(df):
    labels = []
    for i in range(len(df)):
        if i >= len(df)-1: labels.append(-1); continue
        e = df.iloc[i]['close']
        tp, sl = e*(1-SHORT_TP), e*(1+SHORT_SL)
        ht, hs = False, False
        for j in range(i+1, min(i+11, len(df))):
            if df.iloc[j]['low'] <= tp: ht=True; break
            if df.iloc[j]['high'] >= sl: hs=True; break
        if ht: labels.append(1)
        elif hs: labels.append(0)
        else: labels.append(-1)
    return np.array(labels)

def add_bear(df):
    for w in [10,20,50]:
        df[f'price_above_sma{w}'] = (df['close']/df['close'].rolling(w).mean()-1)*100
    df['roc_5'] = df['close'].pct_change(5)*100
    df['roc_10'] = df['close'].pct_change(10)*100
    df['roc_decel'] = df['roc_5'] - df['roc_10']
    df['high_rej'] = (df['high']-df['close'])/(df['high']-df['low']+1e-10)
    df['dist_high20'] = (df['close']/df['high'].rolling(20).max()-1)*100
    df['dist_high50'] = (df['close']/df['high'].rolling(50).max()-1)*100
    df['daily_range'] = (df['high']-df['low'])/df['close']
    df['vol_regime'] = (df['daily_range'].rolling(5).mean()/df['daily_range'].rolling(20).mean()).fillna(1)
    df['dist_sma20'] = df['close']/df['close'].rolling(20).mean()-1
    df['dist_sma50'] = df['close']/df['close'].rolling(50).mean()-1
    df['trend_sc'] = (df['high']>df['high'].shift(1)).rolling(5).sum()-(df['low']<df['low'].shift(1)).rolling(5).sum()
    return df

def train():
    logger.info(f"DOGE SHORT (TP={SHORT_TP:.0%} SL={SHORT_SL:.0%})")
    df = pd.read_csv(DATA_DIR / 'doge_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = add_bear(df)
    df['label'] = create_labels(df)

    with open(BASE_DIR / 'required_features.json') as f:
        base = json.load(f)
    bear = ['price_above_sma10','price_above_sma20','price_above_sma50','roc_5','roc_10',
            'roc_decel','high_rej','dist_high20','dist_high50','vol_regime','dist_sma20',
            'dist_sma50','trend_sc','daily_range']
    fc = base + [f for f in bear if f in df.columns and f not in base]
    FD = len(fc)
    logger.info(f"Features: {FD}")

    n1=(df['label']==1).sum(); n0=(df['label']==0).sum()
    logger.info(f"Labels: SHORT={n1} NO={n0} ratio={n1/(n0+1):.2f}")

    train_df = df[df['date']<'2025-01-01']
    val_df = df[(df['date']>='2025-01-01')&(df['date']<'2025-07-01')]
    test_df = df[df['date']>='2025-07-01']

    scaler = RobustScaler()
    Xtr = np.clip(np.nan_to_num(scaler.fit_transform(train_df[fc].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float32)),nan=0,posinf=0,neginf=0),-5,5)
    Xv = np.clip(np.nan_to_num(scaler.transform(val_df[fc].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float32)),nan=0,posinf=0,neginf=0),-5,5)
    Xt = np.clip(np.nan_to_num(scaler.transform(test_df[fc].fillna(0).replace([np.inf,-np.inf],0).values.astype(np.float32)),nan=0,posinf=0,neginf=0),-5,5)

    def seq(X,y):
        s,l=[],[]
        for i in range(SEQ,len(X)):
            if y[i]!=-1: s.append(X[i-SEQ:i]); l.append(y[i])
        return np.array(s) if s else np.zeros((0,SEQ,FD),dtype=np.float32), np.array(l) if l else np.zeros(0,dtype=np.int64)

    ts,tl = seq(Xtr, train_df['label'].values)
    vs,vl = seq(Xv, val_df['label'].values)
    es,el = seq(Xt, test_df['label'].values)

    n0t,n1t = (tl==0).sum(),(tl==1).sum()
    logger.info(f"Seqs: train={len(ts)} val={len(vs)} test={len(es)}")

    aX = np.concatenate([ts]+[ts+np.random.normal(0,0.015,ts.shape).astype(np.float32) for _ in range(2)])
    ay = np.concatenate([tl]*3)

    w0=len(tl)/(2*n0t) if n0t>0 else 1; w1=len(tl)/(2*n1t) if n1t>0 else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNDirectionModel(feature_dim=FD, sequence_length=SEQ, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([w0,w1]).to(device), label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0015, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    tloader = DataLoader(TensorDataset(torch.FloatTensor(aX),torch.LongTensor(ay)),batch_size=64,shuffle=True)
    vloader = DataLoader(TensorDataset(torch.FloatTensor(vs),torch.LongTensor(vl)),batch_size=64)
    eloader = DataLoader(TensorDataset(torch.FloatTensor(es),torch.LongTensor(el)),batch_size=64)

    best,p,be = 0,0,0
    for ep in range(200):
        model.train()
        for xb,yb in tloader:
            xb,yb=xb.to(device),yb.to(device)
            optimizer.zero_grad(); criterion(model(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5); optimizer.step()
        scheduler.step(ep)
        model.eval()
        c,t,ns,nn_=0,0,0,0
        with torch.no_grad():
            for xb,yb in vloader:
                xb,yb=xb.to(device),yb.to(device)
                pred=model(xb).argmax(1); c+=(pred==yb).sum().item(); t+=yb.size(0)
                ns+=(pred==1).sum().item(); nn_+=(pred==0).sum().item()
        acc=c/t if t>0 else 0
        if acc>best and ns>=3 and nn_>=3:
            best=acc; be=ep+1; p=0
            torch.save({'model_state_dict':model.state_dict(),'feature_dim':FD,'sequence_length':SEQ,
                        'model_type':'cnn','short_tp_pct':SHORT_TP,'short_sl_pct':SHORT_SL,
                        'epoch':ep+1,'val_acc':acc}, MODEL_DIR/'DOGE_short_model.pt')
        else: p+=1
        if (ep+1)%20==0: logger.info(f"E{ep+1}: acc={acc:.3f} S={ns} N={nn_} best={best:.3f}")
        if p>=35: break

    if not (MODEL_DIR/'DOGE_short_model.pt').exists():
        logger.error("No model saved"); return

    ckpt=torch.load(MODEL_DIR/'DOGE_short_model.pt',map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    ad,ac,at=[],[],[]
    with torch.no_grad():
        for xb,yb in eloader:
            d,c=model.predict_direction(xb.to(device))
            ad.extend(d.cpu().numpy()); ac.extend(c.cpu().numpy()); at.extend(yb.numpy())
    ad,ac,at=np.array(ad),np.array(ac),np.array(at)

    logger.info(f"\nDOGE SHORT | Best: E{be} acc={best:.3f}")
    logger.info(f"TEST (2025 H2):")
    for th in [0.50,0.55,0.60,0.65,0.70]:
        m=(ad==1)&(ac>=th)
        if m.sum()>0: logger.info(f"  >={th:.0%}: {m.sum()} sig, WR={at[m].mean()*100:.1f}%")

    joblib.dump(scaler, MODEL_DIR/'feature_scaler_short.joblib')
    with open(MODEL_DIR/'short_features.json','w') as f: json.dump(fc,f)

if __name__=="__main__": train()
