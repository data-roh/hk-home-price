"""Synthetic software-validation fixtures. These are NOT Bloomberg observations."""
from pathlib import Path
import numpy as np
import pandas as pd

def synthetic_data(seed=20261017):
    rng=np.random.default_rng(seed);ix=pd.date_range('1997-02-01','2026-09-01',freq='MS');n=len(ix);t=np.arange(n)
    def ar(sd,rho=.8):
        x=np.zeros(n)
        for k in range(1,n):x[k]=rho*x[k-1]+rng.normal(0,sd)
        return x
    fed=np.clip(2.8+1.5*np.sin(t/30)+ar(.15,.9),.05,7)
    hsi=16000+25*t+ar(850,.88)
    sales=np.maximum(600,4300+900*np.sin(2*np.pi*t/12)+ar(450,.5))
    rent=95+.20*t+ar(1.0,.85)
    fx=1.13+.045*np.sin(t/45)+ar(.004,.9)
    cap=24000+40*t+ar(900,.85)
    vacancy=np.clip(8+2*np.sin(t/28)+ar(.4,.7),1,20)
    inflation=2+.8*np.sin(t/20)+ar(.3,.5)
    unemp=np.clip(3.8+.6*np.sin(t/22)+ar(.15,.7),1,9)
    error=ar(2.0,.72)
    price=70-3.5*fed+.0030*hsi+.0012*sales+.75*rent+22*fx+error
    eps=rng.normal(0,.8,n);u=np.zeros(n)
    for j in range(2,n):u[j]=.6*u[j-1]-.18*u[j-2]+eps[j]+.25*eps[j-1]
    office=25+.0015*cap-1.15*vacancy+.35*inflation-.8*unemp-.6*fed+u
    return pd.DataFrame(dict(home_price=price,fed_rate=fed,hsi=hsi,secondary_sales=sales,home_rent=rent,cnyhkd=fx,office_rent=office,market_cap=cap,office_vacancy=vacancy,inflation_yoy=inflation,unemployment=unemp),index=ix).rename_axis('date')

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'data/demo';out.mkdir(parents=True,exist_ok=True);synthetic_data().to_csv(out/'synthetic_monthly_data.csv')
