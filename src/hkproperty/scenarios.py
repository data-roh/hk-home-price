import numpy as np
import pandas as pd

PERCENT_UNITS={'fed_rate','office_vacancy','inflation_yoy','unemployment'}
POSITIVE={'home_price','hsi','secondary_sales','home_rent','cnyhkd','office_rent','market_cap'}

def baseline_path(history,features,horizon=12,method='last'):
    ix=pd.date_range(history.index[-1]+pd.offsets.MonthBegin(1),periods=horizon,freq='MS')
    path=pd.DataFrame(index=ix,columns=list(features),dtype=float)
    for col in features:
        if method=='last':path[col]=history[col].iloc[-1]
        elif method=='seasonal_naive':
            values=list(history[col].astype(float))
            if len(values)<12:raise ValueError('Need 12 months for seasonal-naive X paths')
            for t in range(horizon):values.append(values[-12])
            path[col]=values[-horizon:]
        else:raise ValueError('Unknown exogenous baseline path')
    return path.rename_axis('date')

def apply_shocks(path,shocks,shape='permanent',half_life=3):
    z=path.copy();h=np.arange(1,len(z)+1)
    if shape=='permanent':weight=np.ones(len(z))
    elif shape=='ramp':weight=h/len(z)
    elif shape=='temporary':weight=np.exp(-np.log(2)*(h-1)/half_life)
    else:raise ValueError('Unknown shock shape')
    for col,shock in shocks.items():
        if col not in z:raise ValueError('Shock variable absent: '+col)
        # Rate inputs: percentage points. Other levels: percentage changes.
        if col in PERCENT_UNITS:z[col]=z[col]+float(shock)*weight
        else:z[col]=z[col]*(1+float(shock)/100*weight)
    if not np.isfinite(z.to_numpy()).all():raise ValueError('Non-finite scenario inputs')
    for col in z:
        if col in POSITIVE and (z[col]<=0).any():raise ValueError('Scenario requires positive '+col)
        if col in {'office_vacancy','unemployment'} and ((z[col]<0)|(z[col]>100)).any():raise ValueError('Percentage out of 0–100 for '+col)
    return z

def extrapolation_report(train,future):
    out=[]
    for col in future:
        lo,hi=train[col].min(),train[col].max();outside=(future[col]<lo)|(future[col]>hi)
        out.append(dict(series=col,train_min=lo,train_max=hi,scenario_min=future[col].min(),scenario_max=future[col].max(),outside_training_range=bool(outside.any())))
    return pd.DataFrame(out)
