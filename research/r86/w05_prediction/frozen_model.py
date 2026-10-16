#!/usr/bin/env python3
"""R86 W05 chronological models M0 identity, M1 repaired 09:25, M2 sector
relative auction, M3 sector+virtual 0924. Freeze training through June23,
embargo 3 full sessions, never fit scaler/threshold on post-train. 3d
recapture label != realized trading profit. Historical folds previously seen.
"""
from pathlib import Path
import pandas as pd,numpy as np,json
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
R=Path(__file__).resolve().parents[1]
L=Path.home()/"Documents/duanlong_relaunch_research/leader_relaunch_lifecycle"
O=R/"w05_prediction"
E=pd.read_csv(L/"r84_auction_retake_20261016/w07_shadow_oct/R84_R83_RESET_589_auction_and_future_label_separated.csv",dtype={"code":str})
P=pd.read_csv(R/"w02_sector_peers/R86_auction_relative_peer_features_every_reset.csv",dtype={"code":str})
C=pd.read_parquet(R/"w01_canonical/R86_REPAIRED_FULL_MARKET_0925.parquet")
C.code=C.code.astype(str).str.zfill(6)
K=pd.read_csv(L/"r84_auction_retake_20261016/w02_server_virtual/R84_server_7709_targeted_prior_0925_snapshot_features.csv",dtype={"code":str})
assert len(E)==589 and len(P)==896
cols=["buy_date","code","sector_eligible_peers","sector_gap_lead_pp",
 "sector_peer_positive_fraction","sector_open_gap_percentile","sector_peer_median_ratio10",
 "own_R21_theme_captured_at_trade"]
E=E.merge(P[cols].rename(columns={"buy_date":"auction_day"}),
 on=["auction_day","code"],how="left",validate="m:1")
E=E.merge(C[["date","code","r86_auction_price","r86_auction_amount_yuan",
 "r86_lag10_ratio","r86_source_type"]].rename(columns={"date":"auction_day"}),
 on=["auction_day","code"],how="left",validate="m:1")
E=E.merge(K[["date","code","first_0920_px","last_0924_px",
 "first_0920_matched","last_0924_matched","last_0924_unmatched",
 "last_0924_direction","last_0924_clock"]].rename(columns={"date":"auction_day"}),
 on=["auction_day","code"],how="left",validate="m:1")
E["r86_open_gap_pct"]=100*(E.r86_auction_price/E.reset_close-1)
E["r86_log_amount"]=np.log1p(E.r86_auction_amount_yuan.clip(lower=0))
E["r86_log_lagratio"]=np.log(E.r86_lag10_ratio.clip(lower=.01))
E["theme_n_peers"]=E.sector_eligible_peers
E["sector_known"]=E.sector_eligible_peers.ge(4).astype(int)
E["r86_virtual_late_lift"]=(E.last_0924_px/E.first_0920_px-1).clip(-.2,.2)
E["r86_virtual_match_ratio"]=np.log((E.last_0924_matched+1)/(E.first_0920_matched+1))
E["r86_virtual_unmatched_signed"]=(E.last_0924_direction*E.last_0924_unmatched/(E.last_0924_matched+1)).clip(-20,20)
E["r86_virtual_clock_valid"]=E.last_0924_clock.notna().astype(int)
E["label"]=E.future_reclaim_within3.astype(int)
E["period"]=np.select([E.reset_date.le("2026-06-23"),E.reset_date.le("2026-06-30"),
 E.reset_date.le("2026-08-31"),E.reset_date.le("2026-09-23")],
 ["TRAIN_JAN_JUN23","EMBARGO","JUL_AUG","SEP_TO23"],default="OCT_METADATA_ONLY")
F0=["max_drop_pct","streak_prior20_max","since_board","prior_successful_recaptures"]
F1=F0+["r86_open_gap_pct","r86_log_amount","r86_log_lagratio"]
F2=F1+["sector_gap_lead_pp","sector_peer_positive_fraction","sector_open_gap_percentile",
 "sector_peer_median_ratio10","sector_known","theme_n_peers"]
F3=F2+["r86_virtual_late_lift","r86_virtual_match_ratio","r86_virtual_unmatched_signed","r86_virtual_clock_valid"]
base=E[E.r86_auction_price.notna()&~E.censored].copy()
train=base[base.period.eq("TRAIN_JAN_JUN23")].copy()
assert train.label.nunique()==2 and len(train)>150
print("R86_M0M3_TRAIN",len(train),"positive",int(train.label.sum()),flush=True)
predict=base[["code","reset_date","auction_day","period","label","r86_open_gap_pct",
 "r86_lag10_ratio","sector_eligible_peers","own_R21_theme_captured_at_trade",
 "sector_gap_lead_pp"]].copy()
results=[];models={}
for name,cols in [("M0_PRIOR_CORE",F0),("M1_REPAIRED_FINAL0925",F1),
 ("M2_PLUS_PEER_RELATIVE",F2),("M3_PLUS_VIRTUAL0924",F3)]:
 clf=Pipeline([("fill",SimpleImputer(strategy="median",add_indicator=True)),
  ("std",StandardScaler()),("clf",LogisticRegression(C=.3,penalty="l2",solver="liblinear",max_iter=2000))])
 clf.fit(train[cols],train.label)
 trainp=clf.predict_proba(train[cols])[:,1]
 threshold=float(np.quantile(trainp,.80))
 preds=clf.predict_proba(base[cols])[:,1]
 predict[name]=preds
 predict[name+"_ALERT"]=preds>=threshold
 models[name]={"features":cols,"train_n":len(train),"train_successes":int(train.label.sum()),
     "train_top20pct_threshold":threshold}
 for fold in ["TRAIN_JAN_JUN23","JUL_AUG","SEP_TO23","OCT_METADATA_ONLY"]:
  z=base[base.period.eq(fold)]
  if len(z)==0:continue
  ind=(base.period==fold).values
  ys=z.label.values;ps=preds[ind]
  win=(ps>=threshold);nt=int(win.sum())
  obj={"model":name,"period":fold,"n":len(z),"eventual_recaptures":int(ys.sum()),
   "baseline_recap_pct":100*ys.mean(),
   "AUC":roc_auc_score(ys,ps) if len(set(ys))==2 else np.nan,
   "AP":average_precision_score(ys,ps) if len(set(ys))==2 else np.nan,
   "Brier":brier_score_loss(ys,ps),
   "top20_train_threshold":threshold,"alerts":nt,"correct_alerts":int(ys[win].sum()),
   "alert_recap_pct":100*ys[win].mean() if nt else np.nan,
   "sector_certified_capture_count":int(z.own_R21_theme_captured_at_trade.sum()),
   "sector_peer4_count":int(z.sector_eligible_peers.ge(4).sum())}
  results.append(obj)
  print("R86_ML_FOLD",obj,flush=True)
res=pd.DataFrame(results)
res.to_csv(O/"R86_M0_M3_CORRECTED_0925_TIME_FORWARD_SCORES.csv",index=False)
predict.to_csv(O/"R86_R83_589_EVENT_ASOF_SCORES_AVAILABLE_AUCTION.csv",index=False)
qa={"status":"R86_4_MODELS_CORRECTED_TDX_AND_PEER_INFO_TIMEBLOCKS",
 "reset_label_total":len(E),"auction_valid_completed_events":len(base),
 "train_through":"2026-06-23","embargo_through":"2026-06-30",
 "train_n":len(train),"models":models,"historical_auction_backfilled":True,
 "training_scaler_imputer_threshold_never_fitted_on_later":True,
 "outcomes_future_used_only_as_label":True,
 "true_OOS":False,"sector_R21_reconstructed_not_proven_live":True,
 "Oct_R21_theme_missing_must_be_flagged":True,
 "profit_not_equivalent_to_recapture_label":True,
 "R50_untouched":True}
(O/"qa.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2))
print("R86_W05_QA",json.dumps(qa,ensure_ascii=False),flush=True)