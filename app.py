# -*- coding: utf-8 -*-
"""BétonLab : essais de compression, statistiques et modèles pédagogiques.

Une ligne représente une éprouvette. Les essais fictifs restent dans la session.
La connexion PostgreSQL provient de l'environnement, des secrets ou de DPAPI.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from hashlib import sha256
from html import escape
from io import BytesIO, StringIO
import json
import math
import os
import secrets
import uuid
import warnings
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg
from psycopg.rows import dict_row
from scipy import stats
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

VERSION = '1.0'
CYAN, INK, ROSE, GREEN, GOLD = '#006E87', '#23272A', '#AE345D', '#217257', '#9C7600'
FIELDS = ['reference','chantier','ouvrage','lot','date_essai','age_jours','forme','resistance_mpa']
GROUP = ['chantier','ouvrage','lot','age_jours','forme']
FORMS = ['Cylindre','Cube']
SCHEMA_SQL = '''CREATE TABLE IF NOT EXISTS public.betonlab_essais (
 id uuid PRIMARY KEY,
 reference varchar(64) NOT NULL,
 chantier varchar(80) NOT NULL,
 ouvrage varchar(80) NOT NULL,
 lot varchar(80) NOT NULL,
 date_essai date NOT NULL,
 age_jours integer NOT NULL CHECK (age_jours BETWEEN 1 AND 3650),
 forme varchar(16) NOT NULL CHECK (forme IN ('Cylindre','Cube')),
 resistance_mpa double precision NOT NULL CHECK (resistance_mpa BETWEEN 0.1 AND 250),
 created_at timestamptz NOT NULL DEFAULT now(),
 UNIQUE (chantier,ouvrage,lot,age_jours,forme,reference)
)'''
INDEX_SQL = '''CREATE INDEX IF NOT EXISTS betonlab_groupe_date_idx
 ON public.betonlab_essais (chantier,ouvrage,lot,age_jours,forme,date_essai)'''


def today():
    return datetime.now(ZoneInfo('Europe/Paris')).date()


def number(value, name, minimum=None, maximum=None):
    if isinstance(value,(bool,np.bool_)):
        raise ValueError(name+' : nombre attendu.')
    try:
        v=float(value)
    except (TypeError,ValueError):
        raise ValueError(name+' : valeur manquante ou non numérique.') from None
    if not math.isfinite(v) or minimum is not None and v<minimum or maximum is not None and v>maximum:
        raise ValueError(f'{name} : valeur hors des bornes autorisées.')
    return v


def validate_trial(raw):
    clean={}
    for key in ['reference','chantier','ouvrage','lot']:
        value=raw.get(key)
        limit=64 if key=='reference' else 80
        if not isinstance(value,str) or not value.strip() or len(value.strip())>limit or any(ord(c)<32 for c in value):
            raise ValueError(f'{key} : renseigner de 1 à {limit} caractères sans retour à la ligne.')
        clean[key]=value.strip()
    try:
        value=raw.get('date_essai')
        if isinstance(value,datetime): value=value.date()
        if not isinstance(value,date): value=date.fromisoformat(str(value))
    except (TypeError,ValueError):
        raise ValueError('date_essai : utiliser AAAA-MM-JJ.') from None
    if not date(1900,1,1)<=value<=today():
        raise ValueError('La date doit être comprise entre 1900 et aujourd’hui.')
    clean['date_essai']=value
    age=number(raw.get('age_jours'),'Âge',1,3650)
    if not age.is_integer(): raise ValueError('L’âge doit être un nombre entier de jours.')
    clean['age_jours']=int(age)
    form=raw.get('forme','Cylindre')
    if form not in FORMS: raise ValueError('forme : Cylindre ou Cube attendu.')
    clean['forme']=form
    clean['resistance_mpa']=number(raw.get('resistance_mpa'),'Résistance',.1,250)
    identity=[clean[k].casefold() if isinstance(clean[k],str) else clean[k] for k in [*GROUP,'reference']]
    clean['id']=str(uuid.uuid5(uuid.NAMESPACE_URL,json.dumps(identity,ensure_ascii=False)))
    return clean


def validate_batch(rows):
    if not 1<=len(rows)<=1000: raise ValueError('Un envoi doit contenir de 1 à 1 000 essais.')
    clean=[];seen=set()
    for i,row in enumerate(rows,1):
        try:
            c=validate_trial(row)
        except ValueError as exc:
            raise ValueError(f'Ligne {i} : {exc}') from None
        if c['id'] in seen: raise ValueError(f'Ligne {i} : référence répétée dans le même groupe.')
        seen.add(c['id']);clean.append(c)
    return clean


def read_csv(data):
    if len(data)>5*1024*1024: raise ValueError('CSV limité à 5 Mo.')
    try:
        frame=pd.read_csv(StringIO(data.decode('utf-8-sig')),sep=None,engine='python',dtype=str,nrows=1001)
    except Exception:
        raise ValueError('CSV illisible : utiliser UTF-8 et une virgule ou un point-virgule comme séparateur.') from None
    required=set(FIELDS)-{'forme'}
    if not required.issubset(frame.columns):
        raise ValueError('Colonnes requises : '+', '.join(sorted(required)))
    records=frame.to_dict('records')
    for row in records:
        row['resistance_mpa']=str(row['resistance_mpa']).replace(',','.')
    return validate_batch(records)


def fictive_trials(small=False):
    values=[25,27,29,31,33,35] if small else [28,29,30,31,32]*4+[29,30,31,42]
    return validate_batch([dict(reference=f'FICT-{i+1:02}',chantier='Chantier fictif',ouvrage='Poteaux pédagogiques',
        lot='Calcul à 6 valeurs' if small else 'Lot de démonstration',date_essai=date(2026,1,1)+timedelta(days=i),
        age_jours=28,forme='Cylindre',resistance_mpa=float(x)) for i,x in enumerate(values)])


def sample_values(values):
    try: x=np.asarray(values,dtype=float)
    except (TypeError,ValueError): raise ValueError('Résistances non numériques.') from None
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError('Utiliser une série non vide de résistances positives et finies.')
    return x


def describe(values,threshold):
    x=sample_values(values); t=number(threshold,'Seuil',.1,250)
    sd=float(x.std(ddof=1)) if len(x)>1 else None
    return dict(n=len(x),mean=float(x.mean()),sd=sd,cv=100*sd/x.mean() if sd is not None else None,
        median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()),
        empirical_q05=float(np.quantile(x,.05,method='linear')),below=int(np.count_nonzero(x<t)),
        observed_p=float(np.mean(x<t)),threshold=t)


def fit_models(values):
    x=sample_values(values)
    if len(x)<2 or np.ptp(x)<1e-9:
        raise ValueError('Au moins deux valeurs distinctes sont nécessaires pour ajuster une loi.')
    mu,sigma=stats.norm.fit(x)
    normal=dict(family='Normale',mu=float(mu),sigma=float(sigma),
                aic=float(4-2*np.sum(stats.norm.logpdf(x,mu,sigma))))
    weibull=None
    if len(x)>=5:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                shape,_,scale=stats.weibull_min.fit(x,floc=0)
            if np.isfinite([shape,scale]).all() and shape>0 and scale>0:
                weibull=dict(family='Weibull',shape=float(shape),scale=float(scale),
                    aic=float(4-2*np.sum(stats.weibull_min.logpdf(x,shape,loc=0,scale=scale))))
        except (ValueError,RuntimeError,FloatingPointError):
            weibull=None
    return dict(normal=normal,weibull=weibull)


def law(model):
    if model['family']=='Normale':
        return stats.norm(loc=model['mu'],scale=model['sigma'])
    if model['family']=='Weibull':
        return stats.weibull_min(model['shape'],loc=0,scale=model['scale'])
    raise ValueError('Loi non reconnue.')


def bayes_update(values,mu0,tau0,sigma,threshold):
    x=sample_values(values)
    mu0=number(mu0,'Moyenne a priori',.1,250)
    tau0=number(tau0,'Incertitude a priori',.01,100)
    sigma=number(sigma,'Écart-type supposé connu',.01,100)
    threshold=number(threshold,'Seuil',.1,250)
    variance=1/(1/tau0**2+len(x)/sigma**2)
    mean=variance*(mu0/tau0**2+float(x.sum())/sigma**2)
    sd=math.sqrt(variance); predictive_sd=math.sqrt(sigma**2+variance)
    return dict(mean=mean,sd=sd,low=mean+stats.norm.ppf(.025)*sd,high=mean+stats.norm.ppf(.975)*sd,
                predictive_sd=predictive_sd,p_below=float(stats.norm.cdf(threshold,mean,predictive_sd)),
                mu0=mu0,tau0=tau0,sigma=sigma,n=len(x))


def control_limits(values,baseline):
    x=sample_values(values)
    if isinstance(baseline,bool) or int(baseline)!=baseline or not 10<=baseline<len(x):
        raise ValueError('La référence doit comprendre au moins 10 essais et laisser un essai à surveiller.')
    baseline=int(baseline)
    center=float(x[:baseline].mean()); mr=float(np.abs(np.diff(x[:baseline])).mean())
    if mr<1e-10: raise ValueError('La référence ne présente pas de variation : limites de contrôle non estimables.')
    sigma=mr/1.128
    lower,upper=center-3*sigma,center+3*sigma
    flags=(x<lower)|(x>upper)
    return dict(center=center,mr=mr,sigma=sigma,lower=lower,upper=upper,flags=flags,
                monitoring=np.arange(len(x))>=baseline,baseline=baseline)


def stress_mpa(force_kn,area_mm2):
    return 1000*number(force_kn,'Force',0,1e8)/number(area_mm2,'Surface',.1,1e9)


def monte_carlo(model,s_mean,s_sd,s_family='Lognormale',count=20000,seed=42):
    s_mean=number(s_mean,'Moyenne de sollicitation',0,1000)
    s_sd=number(s_sd,'Écart-type de sollicitation',0,1000)
    if isinstance(count,bool) or int(count)!=count or not 1000<=count<=100000:
        raise ValueError('Choisir de 1 000 à 100 000 tirages.')
    if isinstance(seed,bool) or int(seed)!=seed or not 0<=seed<=2147483647:
        raise ValueError('Graine entière positive ou nulle requise.')
    if s_family not in ('Lognormale','Normale'): raise ValueError('Loi S non reconnue.')
    if s_family=='Lognormale' and s_mean==0 and s_sd>0:
        raise ValueError('Une sollicitation lognormale de moyenne nulle doit aussi avoir un écart-type nul.')
    rng=np.random.default_rng(int(seed))
    resistance=law(model).rvs(size=int(count),random_state=rng)
    if s_sd==0: demand=np.full(int(count),s_mean)
    elif s_family=='Normale': demand=rng.normal(s_mean,s_sd,int(count))
    else:
        log_sd=math.sqrt(math.log1p((s_sd/s_mean)**2))
        demand=rng.lognormal(math.log(s_mean)-log_sd**2/2,log_sd,int(count))
    mask=resistance<demand; hits=int(mask.sum()); pf=hits/int(count)
    interval=stats.binomtest(hits,int(count)).proportion_ci(.95,method='exact')
    beta=float(-stats.norm.ppf(pf)) if 0<pf<1 else None
    exact=None
    if model['family']=='Normale' and s_family=='Normale':
        exact=float(stats.norm.cdf((s_mean-model['mu'])/math.sqrt(model['sigma']**2+s_sd**2)))
    elif s_sd==0: exact=float(law(model).cdf(s_mean))
    return dict(pf=pf,beta=beta,hits=hits,count=int(count),seed=int(seed),low=float(interval.low),
                high=float(interval.high),exact=exact,R=resistance,S=demand,g=resistance-demand,
                s_mean=s_mean,s_sd=s_sd,s_family=s_family,r_family=model['family'])


def windows_protect(data,decrypt=False):
    import ctypes
    from ctypes import wintypes
    class Blob(ctypes.Structure):
        _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]
    buffer=ctypes.create_string_buffer(data)
    source=Blob(len(data),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)))
    target=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    func=crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    func.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    func.restype=wintypes.BOOL
    kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if not func(ctypes.byref(source),None,None,None,None,1,ctypes.byref(target)):
        raise OSError('Configuration locale chiffrée indisponible pour cet utilisateur Windows.')
    try: return ctypes.string_at(target.data,target.size)
    finally: kernel.LocalFree(target.data)


@lru_cache(maxsize=1)
def local_config():
    if os.name!='nt': return {}
    path=Path(os.environ.get('BETONLAB_SECRET_FILE',str(Path.home()/'Documents/Codex/private/betonlab.connection.dpapi')))
    try: return json.loads(windows_protect(path.read_bytes(),True).decode('utf-8'))
    except (OSError,ValueError): return {}


def configured_dsn():
    value=os.environ.get('BETONLAB_DATABASE_URL') or os.environ.get('DATABASE_URL')
    if value: return value
    try: value=st.secrets.get('BETONLAB_DATABASE_URL') or st.secrets.get('DATABASE_URL')
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError): value=None
    return value or local_config().get('dsn')


def setting(name,default=''):
    if name in os.environ: return os.environ[name]
    try: value=st.secrets.get(name)
    except (FileNotFoundError,st.errors.StreamlitSecretNotFoundError): value=None
    if value is not None: return str(value)
    if name=='BETONLAB_ALLOW_WRITES' and local_config().get('allow_writes'): return 'true'
    return default


@contextmanager
def connection(dsn):
    if not dsn: raise ValueError('Connexion PostgreSQL non configurée.')
    with psycopg.connect(dsn,connect_timeout=10,row_factory=dict_row,sslmode='require') as conn:
        conn.execute("SET LOCAL statement_timeout = '15s'")
        yield conn


def db_groups(dsn):
    with connection(dsn) as conn:
        return conn.execute('SELECT chantier,ouvrage,lot,age_jours,forme,count(*) AS nombre FROM public.betonlab_essais GROUP BY chantier,ouvrage,lot,age_jours,forme ORDER BY chantier,ouvrage,lot,age_jours,forme').fetchall()


def db_read(dsn,group):
    with connection(dsn) as conn:
        rows=conn.execute('SELECT id,reference,chantier,ouvrage,lot,date_essai,age_jours,forme,resistance_mpa FROM public.betonlab_essais WHERE chantier=%s AND ouvrage=%s AND lot=%s AND age_jours=%s AND forme=%s ORDER BY date_essai,reference LIMIT 20001',tuple(group[k] for k in GROUP)).fetchall()
    if len(rows)>20000: raise ValueError('Groupe supérieur à 20 000 essais : affiner les lots avant analyse. Aucun calcul partiel n’est réalisé.')
    return rows


def db_insert(dsn,rows):
    rows=validate_batch(rows)
    try:
        with connection(dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany('INSERT INTO public.betonlab_essais (id,reference,chantier,ouvrage,lot,date_essai,age_jours,forme,resistance_mpa) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',
                    [(r['id'],*[r[k] for k in FIELDS]) for r in rows])
    except psycopg.errors.UniqueViolation:
        raise ValueError('Référence déjà présente dans le même groupe : aucun essai du lot envoyé n’a été ajouté.') from None
    return len(rows)


def safe_csv(frame):
    clean=frame.copy()
    for key in clean.columns:
        if pd.api.types.is_string_dtype(clean[key].dtype) or pd.api.types.is_object_dtype(clean[key].dtype):
            clean[key]=clean[key].map(lambda v:"'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v)
    return clean.to_csv(index=False).encode('utf-8-sig')


def digest(rows):
    return sha256(json.dumps(rows,sort_keys=True,default=str,ensure_ascii=False).encode()).hexdigest()


def fmt(value,digits=2):
    if value is None: return 'Non estimable'
    return f'{float(value):,.{digits}f}'.replace(',',' ').replace('.',',')


def report_pdf(rows,threshold,origin):
    frame=pd.DataFrame(rows); r=describe(frame.resistance_mpa,threshold)
    out=BytesIO(); styles=getSampleStyleSheet()
    styles['BodyText'].leading=15; styles['BodyText'].spaceAfter=8
    def p(text,style='BodyText'): return Paragraph(escape(str(text)),styles[style])
    first=rows[0]
    story=[p('BétonLab','Title'),p('Synthèse pédagogique des essais de compression','Heading2'),p(origin),
        p(f"Groupe : {first['chantier']} / {first['ouvrage']} / {first['lot']} / {first['age_jours']} jours / {first['forme']}"),
        p('Une ligne représente une éprouvette. Les résultats ne constituent pas une décision de conformité réglementaire.'),Spacer(1,10)]
    items=[['Indicateur','Valeur'],['Nombre d’essais',r['n']],['Moyenne',fmt(r['mean'])+' MPa'],
           ['Écart-type échantillon',fmt(r['sd'])+' MPa'],['Coefficient de variation',fmt(r['cv'])+' %'],
           ['Fractile empirique 5 %',fmt(r['empirical_q05'])+' MPa'],['Seuil choisi',fmt(threshold)+' MPa'],
           ['Valeurs observées sous le seuil',f"{r['below']} / {r['n']}"]]
    if len(rows)>1 and np.ptp(frame.resistance_mpa)>1e-9:
        models=fit_models(frame.resistance_mpa)
        for model in models.values():
            if model:
                items.extend([[model['family']+' : fractile ajusté 5 %',fmt(law(model).ppf(.05))+' MPa'],
                              [model['family']+' : P(R < seuil)',fmt(100*law(model).cdf(threshold))+' %']])
    t=Table([[p(x) for x in row] for row in items],colWidths=[285,195],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEFF2')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#CCD9DB')),('LEFTPADDING',(0,0),(-1,-1),9),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.extend([t,Spacer(1,14),p('Hypothèses et interprétation','Heading2'),
        p('Le fractile empirique utilise une interpolation linéaire. La normale est ajustée par maximum de vraisemblance ; son écart-type diffère de l’écart-type échantillon. La Weibull est ajustée avec une origine fixée à zéro. Un faible nombre d’essais, des réplicats dépendants ou un mélange de lots fragilisent l’inférence.'),
        p('Comparer un fractile estimé à une résistance exigée ne vérifie pas les règles d’acceptation d’une norme. Les autres rubriques possèdent leurs propres hypothèses : variance supposée connue pour Bayes, référence choisie pour la carte, indépendance et mêmes unités pour R et S.'),
        p('Les analyses probabilistes et bayésiennes ne valident pas un ouvrage et ne remplacent ni les essais normalisés ni une étude de structure.')])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8); canvas.drawString(50,24,'BétonLab · Rapport pédagogique · '+origin[:50]);canvas.drawRightString(A4[0]-50,24,str(doc.page))
    SimpleDocTemplate(out,pagesize=A4,leftMargin=50,rightMargin=45,topMargin=42,bottomMargin=45).build(story,onFirstPage=footer,onLaterPages=footer)
    return out.getvalue()


CSS='''<style>
:root{color-scheme:light}.stApp,[data-testid="stAppViewContainer"]{background:#f0f4f5;color:#23272a}
[data-testid="stCaptionContainer"]{opacity:1!important}
[data-testid="stHeader"]{background:#f0f4f5}[data-testid="stMainBlockContainer"]{max-width:1500px;padding:4.6rem 2.1rem 3rem}
[data-testid="stSidebar"]{background:#23272a;border-right:0}
.bl-side{font:800 26px Arial;color:#fff;padding:12px 0 5px}.bl-side small{color:#72d2df}
.bl-side-meta{font-size:11px;color:#c3d1d4;margin-bottom:26px}
[data-testid="stSidebar"] button{background:transparent;border:1px solid #5d6a6d;color:#f1f5f6;min-height:43px}
[data-testid="stSidebar"] button p,[data-testid="stSidebar"] button span{color:#f1f5f6}
.stApp [data-testid="stSidebar"] button[kind="primary"]{background:#006e87!important;border-color:#6cc2d0!important;color:#fff!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stTextInput"] label{color:#c3d1d4!important}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{color:#edf5f6!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#c3d1d4!important}
.bl-head{display:flex;align-items:center;justify-content:space-between;gap:18px;background:#fff;border-left:7px solid #006e87;border-bottom:1px solid #cad9dd;padding:21px 25px;margin-bottom:16px}
.bl-head h1{margin:0;padding:0;font:800 36px/1.15 Arial;letter-spacing:0;color:#23272a;overflow-wrap:anywhere}
.bl-head p{margin:7px 0 0;font-size:14px;color:#50666c}.bl-stamp{font-size:12px;font-weight:700;color:#006e87;border-bottom:3px solid #dfbd45;padding:8px 0}
.bl-context{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:13px;color:#4a6269;margin:5px 0 18px}
.bl-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 24px}
.bl-kpi{padding:16px 18px;border-radius:6px;background:#fff;border-top:4px solid #006e87;min-width:0}
.bl-kpi:nth-child(2){background:#006e87;color:#fff;border-top-color:#72d2df}
.bl-kpi:nth-child(3){background:#e5eedf;border-color:#607d39}.bl-kpi:nth-child(4){background:#f5e7ee;border-color:#ae345d}
.bl-kpi span{display:block;font-size:12px;margin-bottom:8px}.bl-kpi strong{display:block;font-size:27px;line-height:1.25;overflow-wrap:anywhere}.bl-kpi small{display:block;font-size:12px;margin-top:7px;line-height:1.5}
.bl-note{border-left:4px solid #006e87;background:#e0eef2;padding:13px 16px;font-size:14px;margin:10px 0}.bl-note.warn{background:#f5e7ee;border-color:#ae345d}
.bl-foot{border-top:1px solid #cbd8db;margin-top:27px;padding-top:18px;font-size:12px;color:#52666b}
h2{font-size:24px!important;letter-spacing:0!important}h3{font-size:19px!important;letter-spacing:0!important}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p{color:#50666c!important}
[data-testid="stForm"]{background:#fff;border-radius:6px;border:1px solid #cbd8db}
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button{border-radius:5px;min-height:42px}
.stButton button p,.stDownloadButton button p{white-space:normal;overflow-wrap:anywhere}
.stButton button[kind="primary"],[data-testid="stFormSubmitButton"] button{background:#006e87!important;border-color:#006e87!important;color:#fff!important}
.stButton button:disabled{opacity:.45!important;cursor:not-allowed!important}
[data-testid="stButtonGroup"] button[aria-checked="true"]{background:#e0eef2!important;border-color:#006e87!important;color:#005166!important}
@media(max-width:1100px){[data-testid="stMainBlockContainer"]{padding:4.5rem 1.2rem 2rem}.bl-kpi{padding:14px 12px}.bl-kpi strong{font-size:24px}}
@media(max-width:740px){.bl-head{padding:19px;flex-direction:column;align-items:flex-start;gap:7px}.bl-head h1{font-size:30px}.bl-kpis{grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.bl-kpi strong{font-size:23px}[data-testid="stHorizontalBlock"]{flex-wrap:wrap}[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{width:100%!important;min-width:100%!important}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>'''


def tiles(items):
    st.html('<div class="bl-kpis">'+''.join(f'<div class="bl-kpi"><span>{escape(a)}</span><strong>{escape(b)}</strong><small>{escape(c)}</small></div>' for a,b,c in items)+'</div>')


def note(text,warn=False):
    st.html(f'<div class="bl-note {"warn" if warn else ""}">{escape(text)}</div>')


def plot(fig,key,height=360):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=24,r=18,t=90 if len(fig.data)>1 else 28,b=65),
        font=dict(family='Arial',color=INK,size=13),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h',y=1.06,yanchor='bottom',x=0,font=dict(size=12)))
    fig.update_xaxes(automargin=True,gridcolor='#d6e1e4');fig.update_yaxes(automargin=True,gridcolor='#d6e1e4')
    st.plotly_chart(fig,width='stretch',theme=None,key=key,alt=key.replace('_',' '),config={'displaylogo':False,'scrollZoom':False,'responsive':True})


def editor_allowed(mode):
    if mode=='Exemple fictif': return True
    if setting('BETONLAB_ALLOW_WRITES').lower() in ('true','1','yes'): return True
    token=setting('BETONLAB_WRITE_TOKEN')
    return bool(token) and secrets.compare_digest(str(st.session_state.get('editor_token','')),token)


def save_trials(rows,mode,dsn):
    rows=validate_batch(rows)
    if not editor_allowed(mode): raise ValueError('Édition non autorisée pour cette session.')
    if mode=='Exemple fictif':
        old=st.session_state.get('demo') or []
        ids={r['id'] for r in old}
        if any(r['id'] in ids for r in rows): raise ValueError('Référence déjà présente dans cet exemple : aucun ajout effectué.')
        st.session_state.demo=old+rows
        return len(rows)
    return db_insert(dsn,rows)


def summary_page(rows,threshold,origin):
    if rows:
        r=describe([x['resistance_mpa'] for x in rows],threshold)
        tiles([('Éprouvettes',str(r['n']),origin),('Résistance moyenne',fmt(r['mean'])+' MPa','Valeur observée'),
               ('Dispersion',fmt(r['sd'])+' MPa','Écart-type échantillon'),
               ('Sous le seuil',f"{r['below']} / {r['n']}",f'Seuil choisi : {fmt(threshold)} MPa')])
        st.subheader('Résistances du groupe sélectionné')
        fig=go.Figure(go.Scatter(x=list(range(1,len(rows)+1)),y=[r['resistance_mpa'] for r in rows],
            mode='lines+markers',line=dict(color=CYAN),marker=dict(color=CYAN,size=7),name='Essais',
            customdata=[escape(x['reference']) for x in rows],hovertemplate='%{customdata}<br>%{y:.2f} MPa<extra></extra>'))
        fig.add_hline(y=threshold,line_dash='dash',line_color=ROSE)
        fig.update_layout(xaxis_title='Essai (ordre date puis référence)',yaxis_title='Résistance (MPa)')
        plot(fig,'resistances_du_groupe')
        st.caption('La ligne pointillée représente le seuil choisi, pas une limite de contrôle statistique.')
        st.dataframe(pd.DataFrame(rows)[FIELDS].tail(12),hide_index=True,width='stretch',alt='Douze derniers essais du groupe, ou moins')
    else:
        tiles([('Essais sélectionnés','0','Base non alimentée ou groupe absent'),('Unité de résistance','MPa','1 MPa = 1 N/mm²'),
               ('Âges et lots','Séparés','Aucun mélange automatique'),('Exemple de test','24 essais','Fictifs, conservés en session')])
        st.subheader('Le registre des essais est prêt')
        st.markdown('Une ligne représente une éprouvette écrasée en laboratoire. Les analyses porteront sur un seul chantier, ouvrage, lot, âge et type d’éprouvette à la fois.')
        with st.container(horizontal=True):
            if st.button('Saisir un essai',type='primary',icon=':material/add:'):
                st.session_state.page='Collecte';st.rerun()
            if st.button('Explorer les 24 essais fictifs',icon=':material/science:'):
                st.session_state.demo=fictive_trials();st.session_state.pending_mode='Exemple fictif';st.rerun()
        note('Le jeu fictif est distinct du stockage PostgreSQL. Son chargement ne crée aucun essai dans la base en ligne.')
    with st.expander('Cadre de lecture'):
        st.markdown('La résistance varie entre éprouvettes. Une éprouvette n’est pas automatiquement un prélèvement indépendant : plusieurs réplicats peuvent être corrélés. Aucun calcul ne conclut à lui seul à la conformité réglementaire du béton ou à la sécurité d’un ouvrage.')


def collect_page(rows,group,mode,dsn):
    st.header('Collecter les résultats d’essais')
    allowed=editor_allowed(mode)
    if not allowed: note('Mode lecture seule. La saisie en base nécessite une autorisation d’édition configurée par le propriétaire.')
    default=group or dict(chantier='',ouvrage='',lot='',age_jours=28,forme='Cylindre')
    with st.form('trial_form'):
        a,b=st.columns(2)
        chantier=a.text_input('Chantier',default['chantier'],max_chars=80)
        ouvrage=b.text_input('Ouvrage',default['ouvrage'],max_chars=80)
        a,b=st.columns(2)
        lot=a.text_input('Lot / formulation',default['lot'],max_chars=80)
        reference=b.text_input('Référence de l’éprouvette',max_chars=64)
        a,b,c=st.columns(3)
        day=a.date_input('Date de l’essai',today(),min_value=date(1900,1,1),max_value=today())
        age=b.number_input('Âge (jours)',1,3650,int(default['age_jours']))
        form=c.selectbox('Type d’éprouvette',FORMS,index=FORMS.index(default['forme']))
        resistance=st.number_input('Résistance à la compression (MPa)',.1,250.,30.,.1)
        submitted=st.form_submit_button('Enregistrer l’essai' if mode=='Base en ligne' else 'Ajouter à l’exemple en session',
            type='primary',icon=':material/save:',disabled=not allowed)
    if submitted:
        try:
            save_trials([dict(reference=reference,chantier=chantier,ouvrage=ouvrage,lot=lot,date_essai=day,
                age_jours=age,forme=form,resistance_mpa=resistance)],mode,dsn)
            st.session_state.flash='Essai enregistré en PostgreSQL.' if mode=='Base en ligne' else 'Essai ajouté à l’exemple en session.'
            st.rerun()
        except ValueError as exc: st.error(str(exc))
        except psycopg.Error: st.error('Le serveur a refusé l’enregistrement. Vérifier la connexion et les droits du rôle PostgreSQL.')
    st.subheader('Importer un CSV')
    template=pd.DataFrame(fictive_trials(True))[FIELDS]
    st.download_button('Modèle CSV fictif',safe_csv(template),'betonlab-modele-fictif.csv','text/csv',icon=':material/download:')
    uploaded=st.file_uploader('Fichier d’essais CSV',type=['csv'],max_upload_size=5)
    if uploaded:
        try:
            imported=read_csv(uploaded.getvalue())
            st.dataframe(pd.DataFrame(imported)[FIELDS],hide_index=True,width='stretch',alt='Prévisualisation des essais à importer')
            confirmed=st.checkbox('J’ai vérifié ces essais et leur destination : '+mode,key='csv_confirm')
            if st.button('Enregistrer le lot importé',disabled=not (allowed and confirmed),type='primary',icon=':material/upload:'):
                n=save_trials(imported,mode,dsn)
                st.session_state.flash=f'{n} essais enregistrés dans {mode.lower()}.';st.rerun()
        except (ValueError,UnicodeError) as exc: st.error(str(exc))
        except psycopg.Error: st.error('Import refusé par PostgreSQL. Aucun envoi partiel n’est validé.')
    st.caption('Envoi atomique : un doublon dans le même groupe refuse tout le lot. Les données enregistrées ne sont ni corrigées ni supprimées automatiquement.')


@st.cache_data(max_entries=6,show_spinner=False)
def weibull_test(values):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result=stats.goodness_of_fit(stats.weibull_min,np.asarray(values),known_params={'loc':0},
            statistic='ad',n_mc_samples=199,rng=np.random.default_rng(42))
    if not math.isfinite(float(result.pvalue)): raise ValueError('Le test n’a pas convergé.')
    return float(result.statistic),float(result.pvalue)


def analysis_page(rows,threshold):
    st.header('Analyse probabiliste des résistances')
    x=np.asarray([r['resistance_mpa'] for r in rows]);d=describe(x,threshold)
    tiles([('Moyenne observée',fmt(d['mean'])+' MPa',f"n = {d['n']} éprouvettes"),
           ('Écart-type observé',fmt(d['sd'])+' MPa','Estimateur échantillon : n - 1'),
           ('Coefficient de variation',fmt(d['cv'])+' %','Écart-type / moyenne'),
           ('Fractile empirique 5 %',fmt(d['empirical_q05'])+' MPa','Interpolation linéaire, descriptif')])
    try: models=fit_models(x)
    except ValueError as exc: note(str(exc));return
    records=[]
    for model in models.values():
        if model:
            fitted=law(model);q05=float(fitted.ppf(.05))
            records.append({'Modèle':model['family'],'Fractile ajusté 5 % (MPa)':q05,
                'Écart au seuil (MPa)':q05-threshold,'P(R < seuil) (%)':100*float(fitted.cdf(threshold)),
                'AIC':model['aic']})
    st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch',alt='Fractiles et probabilités des modèles normale et Weibull')
    note('Le fractile 5 % ajusté est un estimateur pédagogique de f_ck, pas un contrôle de conformité normalisé. Le type d’éprouvette, le prélèvement, la norme et son plan d’acceptation restent déterminants.',True)
    a,b=st.columns(2);xx=np.linspace(max(-20,float(x.min()-3*x.std())),float(x.max()+3*x.std()),260)
    with a:
        fig=go.Figure(go.Histogram(x=x,histnorm='probability density',nbinsx=min(18,len(x)),name='Essais',marker_color='#9ac9d2'))
        for model,color in [(models['normal'],CYAN),(models['weibull'],ROSE)]:
            if model: fig.add_trace(go.Scatter(x=xx,y=law(model).pdf(xx),name=model['family'],line=dict(color=color,width=3)))
        fig.update_layout(xaxis_title='Résistance (MPa)',yaxis_title='Densité (1/MPa)');plot(fig,'lois_ajustees')
    with b:
        fig=go.Figure(go.Scatter(x=np.sort(x),y=np.arange(1,len(x)+1)/len(x),mode='lines+markers',line_shape='hv',name='Répartition empirique'))
        for model,color in [(models['normal'],CYAN),(models['weibull'],ROSE)]:
            if model: fig.add_trace(go.Scatter(x=xx,y=law(model).cdf(xx),name=model['family'],line=dict(color=color)))
        fig.add_vline(x=threshold,line_dash='dash',line_color=GOLD)
        fig.update_layout(xaxis_title='Résistance (MPa)',yaxis_title='Probabilité cumulée');plot(fig,'repartition_des_resistances')
    st.caption(f"Normale ajustée : moyenne {fmt(models['normal']['mu'])} MPa, écart-type MLE {fmt(models['normal']['sigma'])} MPa. Le MLE emploie n au dénominateur, pas n - 1.")
    with st.expander('Lire les paramètres et le fractile'):
        st.markdown('Le fractile 5 % est la résistance sous laquelle le modèle prévoit 5 % des éprouvettes de cette population. Ce n’est pas une probabilité d’effondrement et ce n’est pas une borne de confiance sur le fractile inconnu.')
        st.latex(r'q_{0.05,normal}=\mu+\Phi^{-1}(0.05)\,\sigma')
        if models['weibull']:
            w=models['weibull']
            st.markdown(f"**Weibull à origine zéro** : forme k = {fmt(w['shape'],4)} (sans unité), échelle λ = {fmt(w['scale'],4)} MPa.")
            st.latex(r'F_R(r)=1-\exp[-(r/\lambda)^k],\quad r\geq0')
    negative=float(law(models['normal']).cdf(0))
    if negative>.001: note(f'La normale attribue {fmt(100*negative,3)} % à des résistances négatives. Cet artefact indique une limite du modèle.',True)
    st.subheader('Diagnostics d’ajustement')
    if 3<=len(x)<=5000:
        shapiro=stats.shapiro(x)
        pvalue=f'{shapiro.pvalue:.2e}'.replace('.',',') if shapiro.pvalue<.0001 else fmt(shapiro.pvalue,4)
        st.markdown(f"**Shapiro-Wilk (normale)** : statistique {fmt(shapiro.statistic,4)}, p-valeur {pvalue}.")
        st.caption('p < 0,05 remet en question le modèle normal ; p ≥ 0,05 ne le prouve pas. Ce test ne vérifie pas la conformité du béton.')
    else: st.caption('Shapiro-Wilk est affiché uniquement pour 3 à 5 000 valeurs.')
    if st.button('Tester l’ajustement Weibull',icon=':material/query_stats:',disabled=not (5<=len(x)<=500)):
        try:
            with st.spinner('Réajustement de 199 échantillons simulés…'):
                statistic,pvalue=weibull_test(tuple(x))
            st.info(f'Anderson-Darling : {fmt(statistic,4)} ; p-valeur Monte-Carlo : {fmt(pvalue,3)} (199 répétitions, origine fixée à zéro).')
        except (ValueError,RuntimeError,FloatingPointError): st.warning('Le test Weibull n’a pas convergé pour ces données.')
    st.caption('Le test réestime les paramètres à chaque répétition. Un K-S classique utilisant les paramètres ajustés sans correction donnerait une p-valeur inadaptée. L’AIC compare les deux modèles proposés ; il ne certifie aucun modèle.')


def bayes_page(rows,threshold):
    st.header('Combiner la formulation prévue et les essais')
    signature=digest(rows)+str(threshold)
    with st.form('bayes_form'):
        a,b,c=st.columns(3)
        mu0=a.number_input('Moyenne a priori μ₀ (MPa)',.1,250.,30.,.5)
        tau0=b.number_input('Incertitude sur μ₀ : τ₀ (MPa)',.01,100.,4.,.1)
        sigma=c.number_input('Dispersion des essais supposée connue σ (MPa)',.01,100.,5.,.1)
        run=st.form_submit_button('Mettre à jour la moyenne',icon=':material/update:',type='primary')
    note('Modèle normal-normal : σ est supposé connu et fixé par l’utilisateur. τ₀ décrit l’incertitude sur la moyenne, pas la variabilité des éprouvettes. La seule résistance de formulation ne suffit pas à déterminer ces deux dispersions.')
    if run:
        try: st.session_state.bayes=(signature,bayes_update([r['resistance_mpa'] for r in rows],mu0,tau0,sigma,threshold))
        except ValueError as exc: st.error(str(exc))
    saved=st.session_state.get('bayes')
    if saved and saved[0]==signature:
        r=saved[1]
        tiles([('Moyenne a posteriori',fmt(r['mean'])+' MPa','Moyenne du paramètre μ'),('Incertitude sur μ',fmt(r['sd'])+' MPa','Écart-type a posteriori'),
            ('Intervalle crédible 95 %',f"{fmt(r['low'])} à {fmt(r['high'])}",'MPa · concerne la moyenne'),
            ('Prochain essai sous le seuil',fmt(100*r['p_below'])+' %','Probabilité prédictive, pas P(μ < seuil)')])
        xx=np.linspace(min(r['mu0']-4*r['tau0'],r['mean']-4*r['sd']),max(r['mu0']+4*r['tau0'],r['mean']+4*r['sd']),250)
        fig=go.Figure()
        for label,mu,sd,color in [('A priori sur μ',r['mu0'],r['tau0'],ROSE),('A posteriori sur μ',r['mean'],r['sd'],CYAN)]:
            fig.add_trace(go.Scatter(x=xx,y=stats.norm.pdf(xx,mu,sd),name=label,line=dict(color=color,width=3)))
        fig.update_layout(xaxis_title='Moyenne μ (MPa)',yaxis_title='Densité du paramètre μ');plot(fig,'mise_a_jour_bayesienne')
        st.caption(f"Écart-type prédictif du prochain essai : {fmt(r['predictive_sd'])} MPa. L’analyse suppose les essais indépendants et une même population.")
    with st.expander('Formules du modèle'):
        st.latex(r'\tau_n^2=(1/\tau_0^2+n/\sigma^2)^{-1},\quad \mu_n=\tau_n^2(\mu_0/\tau_0^2+n\bar{x}/\sigma^2)')
        st.latex(r'R_{nouveau}\mid x\sim\mathcal{N}(\mu_n,\sigma^2+\tau_n^2)')


def control_page(rows):
    st.header('Carte de contrôle des essais individuels')
    if len(rows)<11:
        note('Au moins 11 essais sont nécessaires ici : 10 pour la référence et un pour la surveillance.');return
    if len(rows)==11:
        baseline=10
        st.caption('Référence fixée aux 10 premiers essais ; un essai reste en surveillance.')
    else:
        baseline=st.slider('Nombre de premiers essais de référence',10,min(50,len(rows)-1),min(20,len(rows)-1))
    x=np.array([r['resistance_mpa'] for r in rows]);r=control_limits(x,baseline)
    watch=r['flags']&r['monitoring']
    tiles([('Référence',str(baseline),'Premiers essais du groupe'),('Centre',fmt(r['center'])+' MPa','Moyenne de référence'),
           ('Limites de contrôle',f"{fmt(r['lower'])} à {fmt(r['upper'])}",'MPa · pas des seuils de conformité'),
           ('Signaux en surveillance',str(int(watch.sum())),f'{len(rows)-baseline} essais après la référence')])
    colors_=[ROSE if flag else CYAN if i>=baseline else '#8b999d' for i,flag in enumerate(r['flags'])]
    fig=go.Figure(go.Scatter(x=np.arange(1,len(x)+1),y=x,mode='lines+markers',marker=dict(color=colors_,size=9),line=dict(color='#aabcc1'),name='Essais'))
    for value,name,color in [(r['lower'],'LCL',ROSE),(r['center'],'Centre',GREEN),(r['upper'],'UCL',ROSE)]:
        fig.add_hline(y=value,line_dash='dash',line_color=color)
    fig.add_vrect(x0=.5,x1=baseline+.5,fillcolor='#cad5d7',opacity=.2,line_width=0)
    fig.update_layout(xaxis_title='Numéro de l’essai (ordre date puis référence)',yaxis_title='Résistance (MPa)');plot(fig,'carte_de_controle')
    note('Les limites utilisent les étendues mobiles des seuls essais de référence : moyenne ± 3 × MR̄ / 1,128. Un signal invite à rechercher une cause ; il ne prouve ni une erreur d’essai ni une non-conformité.')
    flags=pd.DataFrame(rows).loc[r['flags'],FIELDS].copy()
    if len(flags):
        st.dataframe(flags,hide_index=True,width='stretch',alt='Essais hors limites de contrôle, référence et surveillance comprises')
    st.caption('En cas de dates identiques, le tri utilise la référence, pas l’heure réelle de laboratoire. La carte est une exploration rétrospective ; une surveillance opérationnelle exige une référence stable et un ordre chronologique fiable.')


def reliability_page(rows):
    st.header('Fiabilité simplifiée : comparer R et S')
    try: models=fit_models([r['resistance_mpa'] for r in rows])
    except ValueError as exc: note(str(exc));return
    with st.form('reliability_form'):
        a,b=st.columns(2)
        r_name=a.selectbox('Modèle de résistance R',[k for k,v in [('Normale',models['normal']),('Weibull',models['weibull'])] if v])
        s_name=b.selectbox('Loi de sollicitation S',['Lognormale','Normale'])
        units=st.radio('Définition de la sollicitation moyenne',['Contrainte en MPa','Force et surface'],horizontal=True)
        a,b=st.columns(2)
        stress=a.number_input('Contrainte moyenne S (MPa)',.001,1000.,20.,1.)
        sd=b.number_input('Écart-type de S (MPa)',0.,1000.,3.,.1)
        a,b=st.columns(2)
        force=a.number_input('Force moyenne de compression (kN)',0.,1e8,400.,10.)
        area=b.number_input('Surface comprimée constante (mm²)',.1,1e9,20000.,100.)
        a,b=st.columns(2)
        count=a.select_slider('Nombre de simulations',[1000,5000,20000,50000,100000],value=20000)
        seed=b.number_input('Graine reproductible',0,2147483647,42,1)
        run=st.form_submit_button('Simuler R et S',type='primary',icon=':material/play_arrow:')
    note('R et S doivent être des contraintes en MPa. En mode force/surface : S = 1 000 × F(kN) / A(mm²). La surface est supposée constante, et l’écart-type de S reste celui saisi en MPa.')
    sig=digest(rows)
    if run:
        try:
            mean=stress_mpa(force,area) if units=='Force et surface' else stress
            model=models['normal'] if r_name=='Normale' else models['weibull']
            st.session_state.mc=(sig,monte_carlo(model,mean,sd,s_name,count,seed))
        except (ValueError,TypeError) as exc: st.error(str(exc))
    saved=st.session_state.get('mc')
    if saved and saved[0]==sig:
        r=saved[1]
        tiles([('Pf simulée',fmt(100*r['pf'])+' %',f"{r['hits']} / {r['count']} tirages"),
               ('Indice β équivalent',fmt(r['beta'],3) if r['beta'] is not None else 'Non fini','β = -Φ⁻¹(Pf)'),
               ('Sollicitation moyenne',fmt(r['s_mean'])+' MPa',f"Loi {r['s_family']}"),
               ('IC exact 95 % sur Pf',f"{fmt(100*r['low'],3)} à {fmt(100*r['high'],3)}",'% · erreur Monte-Carlo uniquement')])
        if r['exact'] is not None: st.caption(f"Vérification analytique sous ces mêmes lois indépendantes : Pf = {fmt(100*r['exact'],4)} %.")
        if r['s_family']=='Normale' and r['s_sd']>0:
            negative_s=float(stats.norm.cdf(0,r['s_mean'],r['s_sd']))
            if negative_s>.001:
                note(f'La loi normale choisie attribue {fmt(100*negative_s,3)} % à S < 0. Aucune troncature n’est appliquée ; envisager une loi lognormale pour une compression strictement positive.',True)
        fig=go.Figure(go.Histogram(x=r['g'],nbinsx=45,name='g = R - S',marker_color=CYAN))
        fig.add_vline(x=0,line_color=ROSE,line_dash='dash')
        fig.update_layout(xaxis_title='Marge g = R - S (MPa)',yaxis_title='Nombre de tirages');plot(fig,'marge_resistance_sollicitation')
        if r['beta'] is None: note('Avec zéro ou tous les dépassements dans cet échantillon, β n’est pas estimable finiment à partir de Pf simulée. Utiliser l’intervalle, pas une conclusion de sécurité.')
        draws=pd.DataFrame({'R_simulee_mpa':r['R'],'S_simulee_mpa':r['S'],'g_mpa':r['g'],'R_inferieur_S':r['g']<0})
        st.download_button('Tirages de fiabilité CSV',safe_csv(draws),'betonlab-fiabilite.csv','text/csv',icon=':material/download:')
        st.caption(f"Résultat enregistré : R {r['r_family']}, S {r['s_family']}, écart-type S = {fmt(r['s_sd'])} MPa, graine {r['seed']}. Valider à nouveau le formulaire après modification.")
    note('R et S sont indépendantes dans ce modèle. La résistance d’une éprouvette n’est pas directement la capacité portante d’un élément réel. Pf désigne l’événement R < S de ce modèle, pas une probabilité certifiée d’effondrement.',True)


def exports_page(rows,threshold,origin):
    st.header('Données, rapports et tests')
    if rows:
        frame=pd.DataFrame(rows)[FIELDS]
        st.dataframe(frame,hide_index=True,width='stretch',alt='Essais du groupe courant')
        st.download_button('Exporter les essais CSV',safe_csv(frame),'betonlab-essais.csv','text/csv',icon=':material/download:')
        key=digest(rows)+str(threshold)+origin
        if st.button('Préparer la synthèse PDF',type='primary',icon=':material/picture_as_pdf:'):
            st.session_state.pdf=(key,report_pdf(rows,threshold,origin))
        saved=st.session_state.get('pdf')
        if saved and saved[0]==key:
            st.download_button('Télécharger la synthèse PDF',saved[1],'betonlab-synthese.pdf','application/pdf',icon=':material/download:')
        st.caption('Le PDF reprend les essais et leur analyse univariée. Les résultats bayésiens et de fiabilité sont conditionnels à leurs formulaires et ne sont pas inclus dans cette synthèse.')
    with st.expander('Tests à reproduire',expanded=not rows):
        st.markdown('''
1. Charger le **lot fictif à 24 essais** : un seul groupe, âge **28 jours**, type **Cylindre**. Les données restent en session. Avec le seuil **25 MPa**, n = **24**, moyenne = **30,5 MPa**, fractile empirique 5 % = **28 MPa**.
2. Dans **Analyse**, comparer les deux ajustements et le seuil. L’écart-type échantillon diffère du paramètre sigma de la normale MLE. Les tests d’ajustement ne donnent pas un verdict de conformité.
3. Dans **Carte de contrôle**, conserver les **20 premiers essais** comme référence : centre = **30 MPa**, MR̄ = **28/19 MPa**. Le dernier essai, **42 MPa**, doit être signalé en surveillance.
4. Pour un calcul manuel, charger le **petit exemple à 6 essais** : 25, 27, 29, 31, 33 et 35 MPa. Moyenne = **30 MPa**, écart-type échantillon = **√14 = 3,741657 MPa**, fractile empirique 5 % = **25,5 MPa**.
5. Sur ces six essais, dans **Bayes**, choisir μ₀ = **30**, τ₀ = **4**, σ = **5 MPa**. Moyenne a posteriori = **30 MPa**, écart-type sur μ = **1,818182 MPa**. L’intervalle crédible concerne la moyenne ; la probabilité prédictive concerne une nouvelle éprouvette.
6. Dans **Fiabilité**, avec la normale ajustée aux six essais et S normale de moyenne **20 MPa**, écart-type **3 MPa**, la référence analytique est Φ(-10 / √(70/6 + 9)). La simulation doit en être proche. La conversion **400 kN / 20 000 mm²** doit donner **20 MPa**.
7. Importer le modèle CSV en **mode fictif**, confirmer la destination et exporter les essais. Un doublon doit refuser l’envoi complet. Un nom vide, une date future ou une résistance non numérique doivent être rejetés.
8. Revenir à **Base en ligne** : le jeu fictif ne doit pas apparaître. Après changement de groupe ou de seuil, les anciens rapports incompatibles ne sont plus téléchargeables.
''')
    with st.expander('Hypothèses et références'):
        st.markdown('**Population :** même chantier, ouvrage, lot, âge et forme. Pas de conversion automatique cube/cylindre ni de correction d’âge. Une ligne correspond à une éprouvette, pas nécessairement à un prélèvement indépendant.')
        st.markdown('**Bayes :** modèle normal-normal avec variance d’observation supposée connue. **Carte :** limites I issues des étendues mobiles de la référence, interprétation rétrospective. **Fiabilité :** indépendance R/S et contraintes homogènes en MPa.')
        st.markdown('[NIST : carte individuelle](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm) · [SciPy : ajustement et test Monte-Carlo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html) · [Psycopg](https://www.psycopg.org/psycopg3/docs/basic/usage.html)')


def main():
    st.set_page_config(page_title='BétonLab · Essais et probabilités',page_icon=':material/science:',layout='wide')
    defaults={'page':'Tableau de bord','mode':'Base en ligne','demo':None,'threshold':25.}
    for key,value in defaults.items():
        if key not in st.session_state: st.session_state[key]=value
    st.html(CSS);dsn=configured_dsn()
    pages=[('Tableau de bord','dashboard'),('Collecte','edit_note'),('Analyse','analytics'),('Bayes','update'),
           ('Carte de contrôle','monitoring'),('Fiabilité','query_stats'),('Données et tests','folder_open')]
    with st.sidebar:
        st.html('<div class="bl-side">BétonLab<small>.</small></div><div class="bl-side-meta">ESSAIS DE COMPRESSION / LABORATOIRE STATISTIQUE</div>')
        for name,icon in pages:
            if st.button(name,icon=f':material/{icon}:',width='stretch',type='primary' if st.session_state.page==name else 'secondary'):
                st.session_state.page=name;st.rerun()
        st.space(20)
        mode=st.selectbox('Source de données',['Base en ligne','Exemple fictif'],key='mode')
        if setting('BETONLAB_WRITE_TOKEN'):
            st.text_input('Clé d’édition',type='password',key='editor_token')
        with st.popover('Exemples fictifs',icon=':material/science:',width='stretch'):
            kind=st.selectbox('Jeu fictif',['Lot à 24 essais','Petit calcul à 6 essais'])
            confirm=st.checkbox('Remplacer uniquement le jeu fictif en session')
            if st.button('Charger ce jeu fictif',disabled=not confirm,icon=':material/restart_alt:'):
                st.session_state.demo=fictive_trials(kind=='Petit calcul à 6 essais')
                # The selector is already instantiated; a callback is handled on the next run.
                st.session_state.pending_mode='Exemple fictif';st.rerun()
        st.caption('v'+VERSION+' · aucune validation réglementaire')
    st.html('<header class="bl-head"><div><h1>BétonLab</h1><p>Résistances du béton · essais, données et probabilités</p></div><div class="bl-stamp">MPa / COMPRESSION</div></header>')
    if st.session_state.get('flash'): st.success(st.session_state.pop('flash'))
    cloud_ok=False;groups=[]
    try:
        if mode=='Base en ligne':
            if dsn:
                groups=db_groups(dsn);cloud_ok=True
            else: note('Stockage en ligne non configuré : renseigner BETONLAB_DATABASE_URL dans l’environnement ou les secrets. Le mode fictif reste disponible.')
        else:
            demo=st.session_state.demo or []
            groups=list({tuple(r[k] for k in GROUP):{k:r[k] for k in GROUP} for r in demo}.values())
        tuples=[tuple(g[k] for k in GROUP) for g in groups]
        selected=st.selectbox('Groupe d’analyse',tuples,format_func=lambda t:f'{t[0]} / {t[1]} / {t[2]} / {t[3]} j / {t[4]}',key='group_'+mode) if groups else None
        group=dict(zip(GROUP,selected)) if selected else None
        if group and mode=='Base en ligne': rows=db_read(dsn,group)
        elif group: rows=sorted([r for r in st.session_state.demo if tuple(r[k] for k in GROUP)==selected],key=lambda r:(r['date_essai'],r['reference']))
        else: rows=[]
    except (psycopg.Error,ValueError):
        rows=[];group=None;note('Lecture de la base indisponible. Aucun jeu fictif n’a été substitué aux données. Vérifier la configuration, les droits et la taille du groupe.',True)
    threshold=st.number_input('Résistance spécifiée / seuil de comparaison (MPa)',.1,250.,key='threshold',step=1.)
    origin='Exemple fictif · session uniquement' if mode=='Exemple fictif' else 'PostgreSQL en ligne' if cloud_ok else 'Connexion non disponible'
    st.html(f'<div class="bl-context"><span>{escape(origin)}</span><span>{len(rows)} éprouvettes sélectionnées · seuil {fmt(threshold)} MPa</span></div>')
    if st.session_state.page=='Tableau de bord': summary_page(rows,threshold,origin)
    elif st.session_state.page=='Collecte': collect_page(rows,group,mode,dsn)
    elif st.session_state.page=='Données et tests': exports_page(rows,threshold,origin)
    elif not rows: note('Sélectionner un groupe enregistré ou charger un exemple fictif avant cette analyse.')
    else:
        try:
            if st.session_state.page=='Analyse': analysis_page(rows,threshold)
            elif st.session_state.page=='Bayes': bayes_page(rows,threshold)
            elif st.session_state.page=='Carte de contrôle': control_page(rows)
            else: reliability_page(rows)
        except (ValueError,TypeError) as exc: st.error(str(exc))
    st.html('<div class="bl-foot">Un essai = une éprouvette · populations séparées · estimations conditionnelles aux hypothèses · outil pédagogique</div>')


if __name__=='__main__':
    if st.session_state.get('pending_mode'):
        st.session_state.mode=st.session_state.pop('pending_mode')
    main()
