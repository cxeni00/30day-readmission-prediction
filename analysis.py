# %% [markdown]
# # 30일 이내 재입원 예측 모델 개발
#
# - 데이터: Diabetes 130-US hospitals (1999–2008, Health Facts database), `data/readmitted_data.csv`
# - 예측 시점: 퇴원 시점(landmark) / 결과: 퇴원 후 30일 이내 재입원(`readmitted == "<30"`)
# - 분석 단위: 입원(encounter) 1건 / 분할 단위: 환자(`patient_nbr`)
# - 비교 모델: Dummy, Logistic regression(기본·spline 버전), Random Forest, LightGBM
#
# 모든 표는 `outputs/tables`, 그림은 `outputs/figs`에 저장된다.
#
# 실행: `QUICK=1`이면 탐색·bootstrap 횟수를 줄인 점검용 실행이다. 보고용 결과는 반드시 `QUICK` 없이 실행한다.

# %%
import json
import os
import platform
import subprocess
import time
import warnings

import lightgbm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import shap
import sklearn
import statsmodels
import statsmodels.api as sm
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, SplineTransformer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.family"] = ["AppleGothic", "DejaVu Sans"]  # 한글 + 유니코드 마이너스
plt.rcParams["axes.unicode_minus"] = False

SEED = 42  # 분할, 교차검증, 모델, 탐색, bootstrap 모두 이 값을 사용
QUICK = os.environ.get("QUICK") == "1"
N_ITER = 3 if QUICK else 50       # 모델당 random search 횟수 (네 모델 동일)
N_BOOT = 30 if QUICK else 1000    # 환자 단위 bootstrap 반복
N_BOOT_SUB = 20 if QUICK else 200 # 하위집단 AUROC 신뢰구간용 bootstrap
ALARM_RATES = [0.10, 0.20]        # 고위험 알람 비율 (개발자료 OOF 예측의 상위 k%)
MIN_EVENTS_SUB = 50               # 하위집단 해석 최소 사건 수

OUT = "outputs_quick" if QUICK else "outputs"
TAB, FIG = f"{OUT}/tables", f"{OUT}/figs"
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
T0 = time.time()


def save_tab(df, name, index=False):
    df.to_csv(f"{TAB}/{name}.csv", index=index, encoding="utf-8-sig")
    return df


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=200)
    plt.close(fig)


# %% [markdown]
# ## 1. 데이터 불러오기
# - 원자료의 결측 표시는 `?`이다. `A1Cresult`, `max_glu_serum`의 `None`은 결측이 아니라 "검사하지 않음"이라는 정보이므로
#   pandas 기본 결측 처리(`None` → NaN)를 끄고 `?`만 결측으로 읽는다.

# %%
raw = pd.read_csv("data/readmitted_data.csv", na_values=["?"], keep_default_na=False,
                  dtype={"payer_code": str})
print(raw.shape, raw.patient_nbr.nunique())

# %% [markdown]
# ## 2. 코호트 정의 (보고서 3.2–3.4)
# 퇴원 시점에 판정 가능한 기준만 사용한다. 퇴원 시점에는 입원 중 사망과 호스피스 퇴원 여부가 이미 확정되어 있다.
#
# | 제외 | 기준 | 이유 |
# |---|---|---|
# | 사망 퇴원 | discharge_disposition_id 11, 19, 20, 21 | 재입원이 발생할 수 없음 |
# | 호스피스 퇴원 | discharge_disposition_id 13, 14 | 완화의료 대상으로 재입원 예방 개입 대상이 아님 |
# | 성별 미상 | gender = Unknown/Invalid | 성별 하위집단 분석 불가, 3건 |

# %%
EXPIRED, HOSPICE = [11, 19, 20, 21], [13, 14]


def cohort_row(step, d):
    return {"단계": step, "입원 건수": len(d), "고유 환자 수": d.patient_nbr.nunique(),
            "30일 재입원": int((d.readmitted == "<30").sum())}


flow = [cohort_row("전체 입원", raw)]
df = raw[~raw.discharge_disposition_id.isin(EXPIRED)]
flow.append(cohort_row("사망 퇴원 제외", df))
df = df[~df.discharge_disposition_id.isin(HOSPICE)]
flow.append(cohort_row("호스피스 퇴원 제외", df))
df = df[df.gender != "Unknown/Invalid"].reset_index(drop=True)
flow.append(cohort_row("성별 미상 제외 = 최종 코호트", df))
flow = save_tab(pd.DataFrame(flow), "3_3_cohort_flow")
flow["제외 건수"] = (-flow["입원 건수"].diff()).fillna(0).astype(int)
print(flow)

fig, ax = plt.subplots(figsize=(7, 6))
ax.axis("off")
ys = np.linspace(0.9, 0.1, len(flow))
for i, (yy, r) in enumerate(zip(ys, flow.to_dict("records"))):
    ax.text(0.35, yy, f"{r['단계']}\n입원 {r['입원 건수']:,}건 / 환자 {r['고유 환자 수']:,}명\n30일 재입원 {r['30일 재입원']:,}건",
            ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="#eef3fb", ec="#3366aa"))
    if i:
        ax.annotate("", xy=(0.35, yy + 0.08), xytext=(0.35, ys[i - 1] - 0.08), arrowprops=dict(arrowstyle="->"))
        ax.text(0.72, (yy + ys[i - 1]) / 2, f"{r['단계'].split(' 제외')[0]} {r['제외 건수']:,}건 제외",
                ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#fbeeee", ec="#aa3333"))
save_fig(fig, "fig1_cohort_flow")

# %% [markdown]
# ## 3. 결과변수와 예측변수 정의 (보고서 2.2, 4.1–4.4)
# - 결과: `<30` → 1, `>30`·`NO` → 0. `>30`도 재입원이지만 30일 창 밖이므로 음성으로 둔다.
# - 제외 변수: `encounter_id`, `patient_nbr`(식별자), `readmitted`(결과), `weight`(결측 97%),
#   `examide`, `citoglipton`(모든 값이 `No`로 정보 없음)
# - ID 코드의 NULL/Not Mapped/Not Available 값은 결측으로 합친다.
# - 진단코드(ICD-9)는 Strack et al. (2014)의 9개 그룹으로 묶는다.
# - 연령은 10세 구간의 순서를 보존하도록 0–9 정수로 변환한다.
# - 수치형 변수는 생리적으로 불가능한 값이 없으므로(모두 횟수·일수) 이상치를 자르지 않는다.

# %%
y = (df.readmitted == "<30").astype(int).to_numpy()
groups = df.patient_nbr.to_numpy()

NULL_CODES = {"admission_type_id": [5, 6, 8], "discharge_disposition_id": [18, 25, 26],
              "admission_source_id": [9, 15, 17, 20, 21]}


def icd9_group(code):
    if pd.isna(code):
        return np.nan
    if code[0] in "VE":
        return "Other"
    v = float(code)
    if 390 <= v <= 459 or int(v) == 785: return "Circulatory"
    if 460 <= v <= 519 or int(v) == 786: return "Respiratory"
    if 520 <= v <= 579 or int(v) == 787: return "Digestive"
    if int(v) == 250: return "Diabetes"
    if 800 <= v <= 999: return "Injury"
    if 710 <= v <= 739: return "Musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788: return "Genitourinary"
    if 140 <= v <= 239: return "Neoplasms"
    return "Other"


X = df.drop(columns=["encounter_id", "patient_nbr", "readmitted", "weight", "examide", "citoglipton"])
X["age"] = X["age"].str[1:].str.split("-").str[0].astype(int) // 10
for c, codes in NULL_CODES.items():
    X[c] = X[c].where(~X[c].isin(codes)).astype("Int64").astype(str).replace("<NA>", np.nan)
for c in ["diag_1", "diag_2", "diag_3"]:
    X[c] = X[c].map(icd9_group)

NUM = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
       "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
CAT = [c for c in X.columns if c not in NUM]
X[CAT] = X[CAT].astype(object).where(X[CAT].notna(), np.nan)
print(len(NUM), "numeric /", len(CAT), "categorical")

# %% [markdown]
# ## 4. 데이터 분할 (보고서 3.5–3.6)
# - 시험자료: StratifiedGroupKFold(5)의 첫 fold → 환자 기준 약 20%
# - 개발자료 내부 교차검증: StratifiedGroupKFold(5, seed=42)로 만든 **fold 목록 하나를 네 모델이 공유**

# %%
outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
dev_idx, test_idx = next(outer.split(X, y, groups))
assert not set(groups[dev_idx]) & set(groups[test_idx]), "환자 중복"
assert len(dev_idx) + len(test_idx) == len(X)

Xd, Xt, yd, yt, gd, gt = X.iloc[dev_idx], X.iloc[test_idx], y[dev_idx], y[test_idx], groups[dev_idx], groups[test_idx]
CV = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xd, yd, gd))
for tr, va in CV:
    assert not set(gd[tr]) & set(gd[va])
print(f"개발 {len(dev_idx):,}건/{len(set(gd)):,}명, 시험 {len(test_idx):,}건/{len(set(gt)):,}명, 환자 중복 검사 통과")


def characteristics(d, yy):
    age = d.age * 10
    return {"입원 건수": f"{len(d):,}", "고유 환자 수": f"{d.patient_nbr.nunique():,}",
            "30일 이내 재입원 수": f"{yy.sum():,}", "사건률": f"{yy.mean():.1%}",
            "환자당 입원 횟수": f"{len(d) / d.patient_nbr.nunique():.2f}",
            "연령 <60": f"{(age < 60).mean():.1%}", "연령 60–79": f"{((age >= 60) & (age < 80)).mean():.1%}",
            "연령 80+": f"{(age >= 80).mean():.1%}", "여성 비율": f"{(d.gender == 'Female').mean():.1%}",
            "응급입원 비율": f"{(d.admission_type_id == 1).mean():.1%}",
            "재원일수 평균(SD)": f"{d.time_in_hospital.mean():.1f} ({d.time_in_hospital.std():.1f})"}


df_age = df.assign(age=X.age)
char = pd.DataFrame({"전체": characteristics(df_age, y), "개발자료": characteristics(df_age.iloc[dev_idx], yd),
                     "시험자료": characteristics(df_age.iloc[test_idx], yt)})
save_tab(char, "3_6_split_characteristics", index=True)
save_tab(pd.DataFrame(characteristics(df_age, y), index=["값"]).T, "3_4_final_cohort", index=True)
char

# %% [markdown]
# ## 5. 탐색적 데이터 분석 (보고서 5장, 개발자료만 사용)

# %%
dev_raw = df.iloc[dev_idx]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
dev_raw.readmitted.value_counts().reindex(["<30", ">30", "NO"]).plot.bar(ax=axes[0], color=["#c0392b", "#7f8c8d", "#95a5a6"])
axes[0].set_title("원자료 readmitted 분포 (개발자료)")
axes[0].set_ylabel("입원 건수")
per_pt = dev_raw.patient_nbr.value_counts().clip(upper=5).value_counts().sort_index()
per_pt.index = [str(i) if i < 5 else "5+" for i in per_pt.index]
per_pt.plot.bar(ax=axes[1], color="#3366aa")
axes[1].set_title("환자별 입원 횟수 (개발자료)")
axes[1].set_xlabel("입원 횟수")
axes[1].set_ylabel("환자 수")
save_fig(fig, "fig_eda_outcome_repeat")

miss = pd.DataFrame({"결측률": Xd.isna().mean()})
miss.loc["weight (제외)"] = dev_raw.weight.isna().mean()
miss["A1C/혈당 '검사 안 함'"] = np.nan
miss.loc["A1Cresult", "A1C/혈당 '검사 안 함'"] = (Xd.A1Cresult == "None").mean()
miss.loc["max_glu_serum", "A1C/혈당 '검사 안 함'"] = (Xd.max_glu_serum == "None").mean()
miss = miss.sort_values("결측률", ascending=False).round(4)
save_tab(miss[miss.결측률 > 0], "5_2_missing_rate", index=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
mr = miss.결측률[miss.결측률 > 0].sort_values()
axes[0].barh(mr.index, mr * 100, color="#7f8c8d")
axes[0].set(xlabel="결측률 (%)", title="변수별 결측률 (개발자료)")
for i, v in enumerate(mr * 100):
    axes[0].text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
ip = pd.Series(np.minimum(Xd.number_inpatient.to_numpy(), 3)).map(lambda v: "3+" if v == 3 else str(v))
ev = pd.Series(yd).groupby(ip.to_numpy()).agg(["mean", "size"]).reindex(["0", "1", "2", "3+"])
axes[1].bar(ev.index, ev["mean"] * 100, color="#2c7fb8")
axes[1].axhline(yd.mean() * 100, color="k", ls=":", lw=0.8)
for i, (m_, n_) in enumerate(zip(ev["mean"], ev["size"])):
    axes[1].text(i, m_ * 100 + 0.4, f"{m_ * 100:.1f}%\n(n={n_:,})", ha="center", fontsize=8)
axes[1].set(xlabel="입원 전 1년 입원 횟수", ylabel="30일 재입원률 (%)", title="과거 입원 횟수별 재입원률 (점선 = 전체)",
            ylim=(0, ev["mean"].max() * 100 * 1.25))
los = pd.DataFrame({"los": Xd.time_in_hospital.to_numpy(), "y": yd})
for v, c, lab in [(0, "#95a5a6", "재입원 없음"), (1, "#c0392b", "30일 재입원")]:
    dist = los[los.y == v].los.value_counts(normalize=True).sort_index() * 100
    axes[2].plot(dist.index, dist.values, "o-", ms=3, color=c, label=lab)
axes[2].set(xlabel="재원일수 (일)", ylabel="비율 (%)", title="결과별 재원일수 분포")
axes[2].legend()
save_fig(fig, "fig_eda_missing_dist")

num_by_y = Xd[NUM].assign(y=yd).groupby("y").agg(["mean", "median"]).T.round(2)
num_by_y.columns = ["재입원 없음", "30일 재입원"]
save_tab(num_by_y, "5_2_numeric_by_outcome", index=True)
cat_rate = pd.concat({c: Xd[c].fillna("Missing").to_frame("level").assign(y=yd).groupby("level").y.agg(["size", "mean"])
                      for c in ["race", "admission_type_id", "discharge_disposition_id", "diag_1", "A1Cresult", "insulin"]})
cat_rate.columns = ["입원 건수", "사건률"]
save_tab(cat_rate.round(4), "5_2_categorical_event_rate", index=True)
print(miss.head(8))

# %% [markdown]
# ## 6. 전처리 Pipeline과 모델 (보고서 4.5, 6장)
# 모든 추정 단계(결측 대체, 범주 축약, 스케일링)는 Pipeline 안에 있어 학습 fold에서만 계산된다.
# - 범주형: 결측 → `Missing` 상수 대체 → OneHot. 빈도 0.5% 미만 범주와 상위 10개 밖의 범주는 `infrequent`로 합치고,
#   시험자료의 미지 범주도 `infrequent`로 보낸다(`handle_unknown="infrequent_if_exist"`).
# - 수치형: Logistic만 RobustScaler, 트리 모델은 그대로.
# - 불균형: 본분석은 처리하지 않는다. class weight와 SMOTE는 예측확률을 부풀려 보정도와 Brier score를 왜곡하므로
#   부록 비교로만 다루고, 불균형은 임계값(알람 비율)으로 다룬다.

# %%
def onehot(**kw):
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, max_categories=11, **kw)


class LGBMEarlyStop(ClassifierMixin, BaseEstimator):
    """학습 fold 안에서 환자 단위로 일부를 떼어 조기종료 검증에 쓰는 LightGBM.
    fit에 환자 ID(groups)가 필요하다. 검증 fold·시험자료는 조기종료에 쓰이지 않는다."""

    def __init__(self, learning_rate=0.05, max_depth=-1, num_leaves=31, min_child_weight=1e-3, subsample=1.0,
                 colsample_bytree=1.0, reg_lambda=0.0, val_size=0.15, patience=50, max_trees=2000, random_state=SEED):
        self.learning_rate, self.max_depth, self.num_leaves = learning_rate, max_depth, num_leaves
        self.min_child_weight, self.subsample, self.colsample_bytree = min_child_weight, subsample, colsample_bytree
        self.reg_lambda, self.val_size, self.patience = reg_lambda, val_size, patience
        self.max_trees, self.random_state = max_trees, random_state

    def fit(self, X, y, groups):
        y = np.asarray(y)
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
                      .split(X, y, groups))
        self.model_ = LGBMClassifier(n_estimators=self.max_trees, learning_rate=self.learning_rate,
                                     max_depth=self.max_depth, num_leaves=self.num_leaves,
                                     min_child_weight=self.min_child_weight, subsample=self.subsample,
                                     subsample_freq=1, colsample_bytree=self.colsample_bytree,
                                     reg_lambda=self.reg_lambda, n_jobs=1, random_state=self.random_state, verbose=-1)
        self.model_.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                        callbacks=[lightgbm.early_stopping(self.patience, verbose=False)])
        self.best_iteration_ = self.model_.best_iteration_
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def preprocessor(scale):
    cat = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")), ("onehot", onehot())])
    return ColumnTransformer([("num", RobustScaler() if scale else "passthrough", NUM), ("cat", cat, CAT)])


# Spline Logistic: 강의 2주차 p31 "모델 ② Logistic — 추가: 연속변수 spline 버전"
# - 연속형 6개: 3차 B-spline, 균등 절점 5개 / 이용 횟수 3개: 0이 대부분이라 분위수 절점이 겹치므로 log1p 후 RobustScaler
SPL = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
CNT = ["number_outpatient", "number_emergency", "number_inpatient"]
assert sorted(SPL + CNT) == sorted(NUM)
spline_pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("spl", SplineTransformer(n_knots=5, degree=3, knots="uniform"), SPL),
        ("cnt", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", RobustScaler())]), CNT),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")), ("onehot", onehot())]), CAT)])),
    ("clf", LogisticRegression(max_iter=3000))])


MODELS = {
    "Dummy": (Pipeline([("pre", preprocessor(False)), ("clf", DummyClassifier(strategy="prior"))]), None),
    "Logistic": (Pipeline([("pre", preprocessor(True)), ("clf", LogisticRegression(max_iter=3000))]),
                 {"clf__C": st.loguniform(1e-3, 1e2)}),
    "Spline Logistic": (spline_pipe, {"clf__C": st.loguniform(1e-3, 1e2)}),
    "Random Forest": (Pipeline([("pre", preprocessor(False)),
                                ("clf", RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=SEED))]),
                      {"clf__max_depth": [6, 8, 10, 12, 16, None], "clf__min_samples_leaf": st.randint(5, 21),
                       "clf__max_features": ["sqrt", 0.1, 0.2, 0.3]}),
    "LightGBM": (Pipeline([("pre", preprocessor(False)),
                           ("clf", LGBMEarlyStop())]),
                 {"clf__learning_rate": st.uniform(0.05, 0.05), "clf__max_depth": [3, 4, 5, 6],
                  "clf__num_leaves": st.randint(7, 64), "clf__min_child_weight": st.loguniform(1e-3, 10),
                  "clf__subsample": st.uniform(0.7, 0.2), "clf__colsample_bytree": st.uniform(0.7, 0.2),
                  "clf__reg_lambda": st.loguniform(1e-3, 10)}),
}

# %% [markdown]
# ### 하이퍼파라미터 탐색 (보고서 6.4)
# - RandomizedSearchCV, 모델당 50회, scoring = AUROC, 공유 fold 목록 `CV`
# - LightGBM은 fold 내부 조기종료: 각 학습 fold에서 환자 단위 15%를 떼어 검증 log loss가 50회 연속 개선되지 않으면 멈춘다
#   (최대 2,000개). 트리 수는 탐색하지 않는다. 검증 fold와 시험자료는 조기종료에 쓰지 않는다.
# - OOF 예측(개발자료)은 알람 임계값을 정하는 데 쓴다.

# %%
best, oof, search_rows, fit_time = {}, {}, [], {}
for name, (pipe, space) in MODELS.items():
    t = time.time()
    fp = {"clf__groups": gd} if name == "LightGBM" else {}  # 조기종료용 환자 ID (fold별로 잘려 전달됨)
    if space is None:
        model = pipe.fit(Xd, yd)
        row = {"최적값": "strategy=prior", "CV AUROC": "0.500"}
    else:
        search = RandomizedSearchCV(pipe, space, n_iter=N_ITER, scoring="roc_auc", cv=CV, n_jobs=-1,
                                    random_state=SEED, refit=True)
        search.fit(Xd, yd, **fp)
        model = search.best_estimator_
        i = search.best_index_
        bp = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}
        if name == "LightGBM":
            bp["조기종료 트리 수"] = int(model.named_steps["clf"].best_iteration_)
        row = {"최적값": json.dumps(bp, ensure_ascii=False),
               "CV AUROC": f"{search.cv_results_['mean_test_score'][i]:.3f} ± {search.cv_results_['std_test_score'][i]:.3f}"}
    fit_time[name] = time.time() - t
    best[name] = model
    oof[name] = cross_val_predict(model, Xd, yd, cv=CV, method="predict_proba", n_jobs=-1, params=fp)[:, 1]
    search_rows.append({"모델": name, "방법": "-" if space is None else "RandomizedSearchCV",
                        "횟수": 0 if space is None else N_ITER,
                        "탐색 범위": "-" if space is None else "; ".join(
                            f"{k.replace('clf__', '')}={v if isinstance(v, list) else f'{v.dist.name}{v.args}'}" for k, v in space.items()),
                        **row, "OOF AUROC": f"{roc_auc_score(yd, oof[name]):.3f}", "시간(초)": round(fit_time[name], 1)})
    print(name, row, f"{fit_time[name]:.0f}s")
search_tab = save_tab(pd.DataFrame(search_rows), "6_4_hyperparameter_search")


def oof_slope(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return sm.Logit(yy, sm.add_constant(np.log(p / (1 - p)))).fit(disp=0).params[1]


# 최종 모델 선정은 개발자료 OOF 예측으로만 한다 (시험자료는 선정 뒤 확인용)
sel_rows = [{"모델": n, "OOF AUROC": round(roc_auc_score(yd, oof[n]), 4),
             "OOF AUPRC": round(average_precision_score(yd, oof[n]), 4), "OOF Brier": round(brier_score_loss(yd, oof[n]), 5),
             "OOF 보정 기울기": round(oof_slope(yd, oof[n]), 3), "시간(초)": round(fit_time[n], 1)}
            for n in best if n != "Dummy"]
save_tab(pd.DataFrame(sel_rows), "6_5_oof_selection")
search_tab

# %% [markdown]
# ## 7. 시험자료 평가 (보고서 7장)
# 시험자료는 여기서 한 번만 사용한다.
# - 알람 임계값: 개발자료 OOF 예측의 상위 10%·20% 지점 → 시험자료에 그대로 적용
# - 신뢰구간: 시험자료를 **환자 단위**로 1,000회 재표본한 percentile CI. 모델 간 차이는 같은 재표본에서 계산한 paired 차이

# %%
prob = {name: m.predict_proba(Xt)[:, 1] for name, m in best.items()}
thr = {name: {r: np.quantile(oof[name], 1 - r) for r in ALARM_RATES} for name in best}


def threshold_metrics(yy, p, t):
    flag = p >= t
    tp = (flag & (yy == 1)).sum()
    return {"민감도": tp / yy.sum(), "PPV": tp / flag.sum() if flag.sum() else np.nan,
            "알람/100명": 100 * flag.mean(), "NNE": flag.sum() / tp if tp else np.nan}


# 환자 단위 bootstrap: 환자를 복원추출하고, 뽑힌 환자의 모든 입원을 포함
rng = np.random.default_rng(SEED)
pt_ids, pt_inv = np.unique(gt, return_inverse=True)
pt_rows = [np.flatnonzero(pt_inv == i) for i in range(len(pt_ids))]
pt_rows = np.array(pt_rows, dtype=object)
METRICS = {"AUROC": roc_auc_score, "AUPRC": average_precision_score, "Brier": brier_score_loss}
boot = {(n, m): [] for n in best for m in METRICS}
for _ in range(N_BOOT):
    idx = np.concatenate(pt_rows[rng.integers(0, len(pt_ids), len(pt_ids))])
    for n in best:
        for m, f in METRICS.items():
            boot[(n, m)].append(f(yt[idx], prob[n][idx]))
boot = {k: np.array(v) for k, v in boot.items()}


def ci(a):
    return np.percentile(a, [2.5, 97.5])


perf_rows = []
for n in best:
    row = {"모델": n}
    for m, f in METRICS.items():
        lo, hi = ci(boot[(n, m)])
        row[f"{m} (95% CI)"] = f"{f(yt, prob[n]):.3f} ({lo:.3f}–{hi:.3f})"
    for r in ALARM_RATES:
        k = int(r * 100)
        if n == "Dummy":  # 모든 입원에 같은 확률을 주므로 상위 k% 알람을 정의할 수 없음
            row.update({f"민감도@상위{k}%": "-", f"PPV@상위{k}%": "-", f"알람/100명@상위{k}%": "-"})
            continue
        tm = threshold_metrics(yt, prob[n], thr[n][r])
        row[f"민감도@상위{k}%"] = round(tm["민감도"], 3)
        row[f"PPV@상위{k}%"] = round(tm["PPV"], 3) if not np.isnan(tm["PPV"]) else np.nan
        row[f"알람/100명@상위{k}%"] = round(tm["알람/100명"], 1)
    row["학습시간(초, 탐색 포함)"] = round(fit_time[n], 1)
    perf_rows.append(row)
perf = save_tab(pd.DataFrame(perf_rows), "7_3_test_performance")
print(f"시험자료 유병률(AUPRC 기준선) = {yt.mean():.3f}")
perf

# %% [markdown]
# ### 보정 기울기와 절편 (보고서 7.3, 10장)
# 시험자료에서 logit(예측확률)에 대한 로지스틱 재보정으로 기울기(이상값 1)를, logit을 offset으로 둔 절편 모형으로
# calibration-in-the-large(이상값 0)를 구한다. 신뢰구간은 환자 단위 bootstrap 200회.

# %%
def logit_p(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_slope(yy, p):
    return sm.Logit(yy, sm.add_constant(logit_p(p))).fit(disp=0).params[1]


def cal_citl(yy, p):
    return sm.GLM(yy, np.ones((len(yy), 1)), family=sm.families.Binomial(), offset=logit_p(p)).fit().params[0]


cal_rows = []
rng_c = np.random.default_rng(SEED)
cal_boot = {n: [] for n in best if n != "Dummy"}
for _ in range(N_BOOT_SUB):
    idx = np.concatenate(pt_rows[rng_c.integers(0, len(pt_ids), len(pt_ids))])
    for n in cal_boot:
        cal_boot[n].append((cal_slope(yt[idx], prob[n][idx]), cal_citl(yt[idx], prob[n][idx])))
for n, bs in cal_boot.items():
    bs = np.array(bs)
    obs, pred = calibration_curve(yt, prob[n], n_bins=10, strategy="quantile")
    cal_rows.append({"모델": n,
                     "보정 기울기 (95% CI)": f"{cal_slope(yt, prob[n]):.3f} ({ci(bs[:, 0])[0]:.3f}–{ci(bs[:, 0])[1]:.3f})",
                     "보정 절편 CITL (95% CI)": f"{cal_citl(yt, prob[n]):.3f} ({ci(bs[:, 1])[0]:.3f}–{ci(bs[:, 1])[1]:.3f})",
                     "최상위 10분위 평균 예측": round(pred[-1], 3), "최상위 10분위 관측": round(obs[-1], 3)})
save_tab(pd.DataFrame(cal_rows), "7_3_calibration")

# %%
diff_rows = []
PAIRS = [(n, "Logistic") for n in ["Spline Logistic", "Random Forest", "LightGBM", "Dummy"]] + [("LightGBM", "Random Forest")]
for a_, b_ in PAIRS:
    for m in ["AUROC", "AUPRC", "Brier"]:
        d = boot[(a_, m)] - boot[(b_, m)]
        lo, hi = ci(d)
        point = METRICS[m](yt, prob[a_]) - METRICS[m](yt, prob[b_])
        diff_rows.append({"비교": f"{a_} − {b_}", "지표": m, "차이": round(point, 4),
                          "95% CI": f"{lo:.4f}–{hi:.4f}", "0 포함": lo <= 0 <= hi})
diff = save_tab(pd.DataFrame(diff_rows), "7_2_paired_bootstrap_diff")
diff

# %%
COLORS = {"Dummy": "#95a5a6", "Logistic": "#2c7fb8", "Spline Logistic": "#9e9ac8", "Random Forest": "#31a354", "LightGBM": "#e6550d"}
fig, ax = plt.subplots(figsize=(5.5, 5))
for n in best:
    fpr, tpr, _ = roc_curve(yt, prob[n])
    ax.plot(fpr, tpr, color=COLORS[n], label=f"{n} ({roc_auc_score(yt, prob[n]):.3f})")
ax.plot([0, 1], [0, 1], "k:", lw=0.8)
ax.set(xlabel="1 − 특이도", ylabel="민감도", title="ROC curve (시험자료)")
ax.legend(loc="lower right")
save_fig(fig, "fig3_roc")

fig, ax = plt.subplots(figsize=(5.5, 5))
for n in best:
    pr, rc, _ = precision_recall_curve(yt, prob[n])
    ax.plot(rc, pr, color=COLORS[n], label=f"{n} ({average_precision_score(yt, prob[n]):.3f})")
ax.axhline(yt.mean(), color="k", ls=":", lw=0.8, label=f"유병률 {yt.mean():.3f}")
ax.set(xlabel="재현율(민감도)", ylabel="정밀도(PPV)", title="Precision–Recall curve (시험자료)")
ax.legend(loc="upper right")
save_fig(fig, "fig4_pr")

fig, ax = plt.subplots(figsize=(5.5, 5))
lim = 0
for n in best:
    if n == "Dummy":
        ax.scatter(prob[n].mean(), yt.mean(), color=COLORS[n], label=n, zorder=3)
        continue
    obs, pred = calibration_curve(yt, prob[n], n_bins=10, strategy="quantile")
    ax.plot(pred, obs, "o-", color=COLORS[n], ms=4, label=f"{n} (Brier {brier_score_loss(yt, prob[n]):.4f})")
    lim = max(lim, pred.max(), obs.max())
lim *= 1.1
ax.plot([0, lim], [0, lim], "k:", lw=0.8)
ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="예측 확률 (10분위 평균)", ylabel="관측 재입원률", title="Calibration curve (시험자료)")
ax.legend(loc="upper left")
save_fig(fig, "fig5_calibration")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
rates = np.linspace(0.01, 0.5, 50)
for n in [k for k in best if k != "Dummy"]:
    tms = [threshold_metrics(yt, prob[n], np.quantile(oof[n], 1 - r)) for r in rates]
    alarms = [t["알람/100명"] for t in tms]
    axes[0].plot(alarms, [t["민감도"] for t in tms], color=COLORS[n], label=n)
    axes[1].plot(alarms, [t["PPV"] for t in tms], color=COLORS[n], label=n)
for a in axes:
    for r in ALARM_RATES:
        a.axvline(100 * r, color="k", ls=":", lw=0.6)
    a.set_xlabel("시험자료 100명당 알람 수")
axes[0].set(ylabel="민감도", title="임계값별 민감도")
axes[1].axhline(yt.mean(), color="k", ls="--", lw=0.6)
axes[1].set(ylabel="양성예측도(PPV)", title="임계값별 PPV")
axes[0].legend()
fig.suptitle("알람 비율(개발자료 OOF 기준 임계값)에 따른 민감도와 PPV")
save_fig(fig, "fig6_threshold")

# %% [markdown]
# ## 8. 모델 해석 (보고서 8장)
# 아래 결과는 **예측 기여도**이며 재입원의 원인이나 치료효과가 아니다.
#
# ### 8.1 Logistic regression 오즈비
# sklearn의 L2 모델은 표준오차를 주지 않으므로, 같은 전처리(기준 범주 1개 제거)로 statsmodels Logit을 개발자료에 다시 적합한다.
# 한 환자의 여러 입원은 독립이 아니므로 **환자 단위 cluster-robust 표준오차**를 쓴다.
# 수치형 변수의 오즈비는 IQR 1단위 증가당이다. 해당 입원이 0.5% 미만인 원핫 열은 완전분리로 추정이 불가하여 이 모형에서만 제외한다.

# %%
pre_sm = ColumnTransformer([("num", RobustScaler(), NUM),
                            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                                              ("onehot", onehot(drop="first", sparse_output=False))]), CAT)])
Zd = pd.DataFrame(pre_sm.fit_transform(Xd), columns=pre_sm.get_feature_names_out())
# 원핫 열 중 해당 입원이 0.5% 미만인 열(희귀 당뇨약의 infrequent 열 등)은 완전분리로 MLE가 발산하므로 제외
Zd = Zd.loc[:, (Zd != 0).mean() >= 0.005]
logit = sm.Logit(yd, sm.add_constant(Zd)).fit(disp=0, maxiter=500, cov_type="cluster", cov_kwds={"groups": gd})
assert logit.mle_retvals["converged"], "Logit 미수렴"
or_tab = pd.DataFrame({"계수": logit.params, "Odds ratio": np.exp(logit.params),
                       "95% CI 하한": np.exp(logit.conf_int()[0]), "95% CI 상한": np.exp(logit.conf_int()[1]),
                       "p": logit.pvalues}).drop("const")
or_tab.index = or_tab.index.str.replace(r"^(num|cat)__", "", regex=True)
or_tab = or_tab.reindex(or_tab.계수.abs().sort_values(ascending=False).index)
save_tab(or_tab.round(4), "8_1_logistic_odds_ratio_all", index=True)
or_top = or_tab[or_tab.p < 0.05].head(10).round(3)

# 최종 L2 모델(sklearn) 자체의 계수. 기준 범주가 없고 규제로 0 쪽으로 수축되어 있어 위 오즈비와 크기는 다르다.
lr_fin = best["Logistic"]
l2 = pd.Series(lr_fin.named_steps["clf"].coef_[0], index=lr_fin.named_steps["pre"].get_feature_names_out())
l2.index = l2.index.str.replace(r"^(num|cat)__", "", regex=True)
l2 = pd.DataFrame({"계수": l2, "exp(계수)": np.exp(l2)}).reindex(l2.abs().sort_values(ascending=False).index)
save_tab(l2.round(4), "8_1_logistic_l2_coef", index=True)
save_tab(or_top, "8_1_logistic_odds_ratio_top", index=True)
or_top

# %% [markdown]
# ### 8.2 Random Forest permutation importance
# 원자료 열 단위로 섞으므로 원핫 열이 아니라 원래 변수 단위의 중요도가 나온다. Gini importance는 사용하지 않는다.

# %%
pi = permutation_importance(best["Random Forest"], Xt, yt, scoring="roc_auc",
                            n_repeats=3 if QUICK else 10, random_state=SEED, n_jobs=-1)
pi_tab = pd.DataFrame({"AUROC 감소 평균": pi.importances_mean, "표준편차": pi.importances_std},
                      index=Xt.columns).sort_values("AUROC 감소 평균", ascending=False)
save_tab(pi_tab.round(5), "8_2_rf_permutation_importance", index=True)
fig, ax = plt.subplots(figsize=(6, 4.5))
top = pi_tab.head(15)[::-1]
ax.barh(top.index, top["AUROC 감소 평균"], xerr=top["표준편차"], color=COLORS["Random Forest"])
ax.set(xlabel="변수를 섞었을 때 AUROC 감소", title="Random Forest permutation importance (시험자료)")
save_fig(fig, "fig8_2_rf_permutation")
pi_tab.head(10)

# %% [markdown]
# ### 8.3 LightGBM SHAP
# 원핫 열의 SHAP 값을 원래 변수별로 합쳐 mean |SHAP|을 계산한다. 대표 환자 2명(고위험 실제 재입원, 저위험 비재입원)의 개별 설명을 함께 제시한다.

# %%
lgb_pre, lgb_clf = best["LightGBM"].named_steps["pre"], best["LightGBM"].named_steps["clf"].model_
n_shap = min(len(Xt), 500 if QUICK else 5000)
shap_idx = np.random.default_rng(SEED).choice(len(Xt), n_shap, replace=False)
Zt = lgb_pre.transform(Xt.iloc[shap_idx])
Zt = Zt.toarray() if hasattr(Zt, "toarray") else Zt
feat = lgb_pre.get_feature_names_out()
sv = shap.TreeExplainer(lgb_clf).shap_values(Zt)
sv = sv[1] if isinstance(sv, list) else sv


def orig_var(f):
    f = f.split("__", 1)[1]
    return f if f in NUM else max((c for c in CAT if f.startswith(c + "_")), key=len)


owner = np.array([orig_var(f) for f in feat])
agg = pd.DataFrame(sv, columns=owner).T.groupby(level=0).sum().T  # 환자별로 변수 단위 SHAP 합
shap_tab = agg.abs().mean().sort_values(ascending=False).to_frame("mean |SHAP| (log-odds)")
save_tab(shap_tab.round(5), "8_3_lgbm_shap_importance", index=True)
fig, ax = plt.subplots(figsize=(6, 4.5))
top = shap_tab.head(15)[::-1]
ax.barh(top.index, top.iloc[:, 0], color=COLORS["LightGBM"])
ax.set(xlabel="mean |SHAP| (log-odds)", title="LightGBM SHAP 중요도 (변수 단위)")
save_fig(fig, "fig8_3_lgbm_shap")

p_s, y_s = prob["LightGBM"][shap_idx], yt[shap_idx]
cases = {"고위험_실제재입원": np.flatnonzero(y_s == 1)[np.argmax(p_s[y_s == 1])],
         "저위험_비재입원": np.flatnonzero(y_s == 0)[np.argmin(p_s[y_s == 0])]}
base = shap.TreeExplainer(lgb_clf).expected_value
base = np.ravel(base)[-1]
for label, i in cases.items():
    e = shap.Explanation(values=agg.iloc[i].to_numpy(), base_values=base, feature_names=list(agg.columns),
                         data=None)
    shap.plots.waterfall(e, max_display=8, show=False)
    plt.title(f"{label.replace('_', ' ')} (예측확률 {p_s[i]:.2f})")  # 크기는 shap 기본값 유지(줄이면 막대 안 숫자가 잘림)
    save_fig(plt.gcf(), f"fig8_3_shap_case_{label}")
shap_tab.head(10)

# %% [markdown]
# ## 9. 하위집단 분석 (보고서 9장)
# 사건이 50건 미만인 집단은 `표본 부족`으로 표시한다. AUROC 95% CI는 하위집단 안에서 환자 단위 bootstrap 200회.

# %%
age_t = Xt.age.to_numpy() * 10
SUB = {"전체": np.ones(len(yt), bool), "여성": (Xt.gender == "Female").to_numpy(), "남성": (Xt.gender == "Male").to_numpy(),
       "연령 <60": age_t < 60, "연령 60–79": (age_t >= 60) & (age_t < 80), "연령 80+": age_t >= 80,
       **{f"인종 {r}": (Xt.race.fillna("Missing") == r).to_numpy() for r in ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Missing"]},
       "응급입원": (Xt.admission_type_id == "1").to_numpy(), "응급 외 입원(결측 포함)": (Xt.admission_type_id != "1").to_numpy(),
       "과거 1년 입원 있음": (Xt.number_inpatient > 0).to_numpy(), "과거 1년 입원 없음": (Xt.number_inpatient == 0).to_numpy()}
sub_rows = []
rng = np.random.default_rng(SEED)
for sname, mask in SUB.items():
    ys, gs = yt[mask], gt[mask]
    row = {"하위집단": sname, "표본": int(mask.sum()), "사건": int(ys.sum()), "사건률": round(ys.mean(), 3),
           "해석": "표본 부족" if ys.sum() < MIN_EVENTS_SUB else ""}
    u, inv = np.unique(gs, return_inverse=True)
    rows_sub = np.array([np.flatnonzero(inv == i) for i in range(len(u))], dtype=object)
    for n in ["Logistic", "Random Forest", "LightGBM"]:
        ps = prob[n][mask]
        bs = []
        for _ in range(N_BOOT_SUB):
            idx = np.concatenate(rows_sub[rng.integers(0, len(u), len(u))])
            if ys[idx].min() != ys[idx].max():
                bs.append(roc_auc_score(ys[idx], ps[idx]))
        lo, hi = ci(boot[(n, "AUROC")]) if sname == "전체" else ci(bs)  # 전체 행은 7.3과 같은 1,000회 CI
        row[f"{n} AUROC (95% CI)"] = f"{roc_auc_score(ys, ps):.3f} ({lo:.3f}–{hi:.3f})"
        row[f"{n} AUPRC"] = round(average_precision_score(ys, ps), 3)
        row[f"{n} Brier"] = round(brier_score_loss(ys, ps), 4)
    sub_rows.append(row)
sub = save_tab(pd.DataFrame(sub_rows), "9_subgroup_performance")
sub

# %% [markdown]
# ## 10. 민감도 분석 (부록)
# ### 10.1 환자별 첫 입원만 사용
# 본분석의 최적 하이퍼파라미터를 그대로 쓰고, 개발자료와 시험자료 각각에서 환자별 첫 입원(encounter_id 최소)만 남겨 다시 학습·평가한다.

# %%
first = df.groupby("patient_nbr").encounter_id.transform("min") == df.encounter_id
fd, ft = first.to_numpy()[dev_idx], first.to_numpy()[test_idx]
sens_rows = []
for n in ["Logistic", "Random Forest", "LightGBM"]:
    m = sklearn.base.clone(best[n]).fit(Xd[fd], yd[fd], **({"clf__groups": gd[fd]} if n == "LightGBM" else {}))
    p = m.predict_proba(Xt[ft])[:, 1]
    sens_rows.append({"모델": n, "개발 입원": int(fd.sum()), "시험 입원": int(ft.sum()), "시험 사건": int(yt[ft].sum()),
                      "AUROC": round(roc_auc_score(yt[ft], p), 3), "AUPRC": round(average_precision_score(yt[ft], p), 3),
                      "Brier": round(brier_score_loss(yt[ft], p), 4),
                      "본분석 AUROC": round(roc_auc_score(yt, prob[n]), 3)})
save_tab(pd.DataFrame(sens_rows), "A_sensitivity_first_encounter")

# %% [markdown]
# ### 10.2 불균형 처리 비교 (Logistic, 개발자료 교차검증)
# 시험자료를 쓰지 않고 공유 fold의 OOF 예측으로 비교한다. SMOTE는 imblearn Pipeline 안에서 학습 fold에만 적용된다.

# %%
C_best = best["Logistic"].named_steps["clf"].C
imb = {"처리 없음": Pipeline([("pre", preprocessor(True)), ("clf", LogisticRegression(C=C_best, max_iter=3000))]),
       "class weight": Pipeline([("pre", preprocessor(True)),
                                 ("clf", LogisticRegression(C=C_best, max_iter=3000, class_weight="balanced"))]),
       "SMOTE": ImbPipeline([("pre", preprocessor(True)), ("smote", SMOTE(random_state=SEED)),
                             ("clf", LogisticRegression(C=C_best, max_iter=3000))])}
imb_rows = []
for k, pipe in imb.items():
    p = cross_val_predict(pipe, Xd, yd, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    imb_rows.append({"방법": k, "OOF AUROC": round(roc_auc_score(yd, p), 3), "OOF AUPRC": round(average_precision_score(yd, p), 3),
                     "OOF Brier": round(brier_score_loss(yd, p), 4), "평균 예측확률": round(p.mean(), 3),
                     "실제 사건률": round(yd.mean(), 3)})
save_tab(pd.DataFrame(imb_rows), "A_imbalance_comparison")

# %% [markdown]
# ## 11. 재현성 기록 (보고서 11장)

# %%
n_feat = len(best["Logistic"].named_steps["pre"].get_feature_names_out())
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain", "analysis.py"], capture_output=True, text=True).stdout.strip():
        commit += " + 커밋되지 않은 변경"
except OSError:
    commit = ""
repro = {"random_seed": SEED, "seed 적용 지점": "시험 분할, 교차검증 fold, 모델(RF·LightGBM), RandomizedSearchCV, bootstrap, SMOTE, SHAP 표본",
         "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
         "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__,
         "imbalanced-learn": imblearn.__version__, "statsmodels": statsmodels.__version__, "matplotlib": matplotlib.__version__,
         "전체 입원 / 고유 환자": f"{len(df):,} / {df.patient_nbr.nunique():,}",
         "사건 수 / 사건률": f"{y.sum():,} / {y.mean():.2%}",
         "EPV (개발자료 사건 수 / 원핫 후 입력 변수 수)": f"{yd.sum():,} / {n_feat} = {yd.sum() / n_feat:.1f}",
         "분할": "StratifiedGroupKFold(5) 첫 fold를 시험자료(약 20%), 그룹 = patient_nbr",
         "교차검증": "StratifiedGroupKFold(5, shuffle, seed 42), 모든 모델이 같은 fold 목록 사용",
         "모델별 탐색": f"RandomizedSearchCV {N_ITER}회 (Dummy 제외), scoring = AUROC",
         "bootstrap": f"시험자료 환자 단위 {N_BOOT}회, percentile CI, paired 차이",
         "코드": f"analysis.py / analysis.ipynb, git commit {commit or '-'}",
         "실행환경": f"{platform.platform()}, CPU {os.cpu_count()}코어",
         "전체 실행시간(분)": round((time.time() - T0) / 60, 1), "QUICK 모드": QUICK}
with open(f"{OUT}/repro.json", "w", encoding="utf-8") as f:
    json.dump(repro, f, ensure_ascii=False, indent=2)

# 부록: 전체 변수 사전
HANDLING = {**{c: "수치형, Logistic만 RobustScaler" for c in NUM},
            **{c: "범주형, 결측→Missing, OneHot(희귀·상위10 밖→infrequent)" for c in CAT}}
var_dict = pd.DataFrame({"자료형": ["수치형" if c in NUM else "범주형" for c in X.columns],
                         "고유값 수": X.nunique().values, "결측률(개발)": Xd.isna().mean().round(4).values,
                         "처리": [HANDLING[c] for c in X.columns]}, index=X.columns)
save_tab(var_dict, "A_variable_dictionary", index=True)
print(json.dumps(repro, ensure_ascii=False, indent=2))
