"""Bounded CP objectives/adaptations and train-only classical fusion."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .cp_transfer_data import SEED
from .cp_transfer_neural import REGRESSION_SETTINGS
from .eyelid_transfer_neural import SETTINGS

INNER_SEED=20261011
OBJECTIVES=('cp_classification','cp_regression','imagenet_control')
STRATEGIES=('A','B','C')
CANDIDATES=tuple(f'{objective}_{strategy}' for objective in OBJECTIVES for strategy in STRATEGIES)
DESIGN={'architecture':'Verified MobileNetV3-Small ImageNet1K_V1; fixed early-prefix maps; dropout0.2; global pooled576 backbone features',
 'public_classification':'CP study Hb<11; adapt binary pediatric head and finaltwo blocks; public-validation AP/patience4 only',
 'public_regression':REGRESSION_SETTINGS,'public_settings':SETTINGS['public'],
 'local_A':'Replace all public heads with new adult Linear576->1; freeze backbone; training-only weighted BCE and inner AP stopping',
 'local_B':'Same new adult head; adapt lasttwo blocks at1e-5, head3e-4; fixed BatchNorm statistics; inner AP stopping',
 'local_C':'Frozen public576 embeddings + exact historical robust18; training-only embedding StandardScaler/PCA8 and fused StandardScaler; L2 logistic C0.1 balanced; no backbone tuning',
 'candidates':list(CANDIDATES),'matched_controls':'Same A/B/C training and preprocessing using ImageNet alone; enables a controlled pretraining contrast beyond earlier differing experiments',
 'local_settings':{k:SETTINGS[k] for k in ('A','B')},'inner_folds':3,'inner_seed':INNER_SEED,'outer_folds':'Exact original119-subject five folds seed20261010',
 'outer_refit_epochs':'Rounded median inner best epochs; fit all outer training only; no outer checkpoint selection',
 'selection':'Highest pooled inner AP across each objective group and allnine; frozen candidate order ties',
 'threshold':'Maximize inner sensitivity subject to specificity>=0.80, then specificity/higher cutoff; fixed0.5 also reported',
 'calibration':'none; weighted pediatric/adult scores not calibrated clinical probabilities',
 'public_target_separation':'Public pediatric and local adult/adolescent labels never pooled; public-only pediatric scores exploratory against adult labels',
 'compute':'CPU one thread, deterministic algorithms, cached original/horizontal-flip early maps; guard checkpoints/resume; no MPS nondeterminism or memory exhaustion',
 'scope':'Exploratory repeated119-person development cohort; public split is curation+near-image grouped, true patient linkage unavailable; no external test or independent clinical validity'}


def select(scores: dict[str,float],objective: str | None=None) -> str:
    eligible=[c for c in CANDIDATES if objective is None or c.startswith(objective+'_')]
    if set(scores)!=set(CANDIDATES) or not np.isfinite(list(scores.values())).all():raise ValueError('Incomplete/nonfinite inner candidate scores')
    return max(eligible,key=lambda name:(scores[name],-CANDIDATES.index(name)))


class FusionProjection(TransformerMixin,BaseEstimator):
    """Scale/project embeddings using training subjects only; preserve raw LAB."""
    def __init__(self,n_components: int=8):self.n_components=n_components
    def fit(self,x: np.ndarray,y: np.ndarray | None=None) -> FusionProjection:
        x=self.check(x)
        if len(x)<=self.n_components:raise ValueError('Insufficient PCA training support')
        self.embedding_scaler_=StandardScaler().fit(x[:,18:]);self.pca_=PCA(self.n_components,svd_solver='full').fit(self.embedding_scaler_.transform(x[:,18:]));self.training_samples_=len(x)
        return self
    def transform(self,x: np.ndarray) -> np.ndarray:
        x=self.check(x);return np.c_[x[:,:18],self.pca_.transform(self.embedding_scaler_.transform(x[:,18:]))]
    @staticmethod
    def check(x: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=594 or not np.isfinite(x).all():raise ValueError('Expected finite robust18+576 embedding features')
        return x


def fusion_pipeline() -> Pipeline:
    return Pipeline([('projection',FusionProjection()),('scaler',StandardScaler()),('classifier',LogisticRegression(C=.1,penalty='l2',class_weight='balanced',solver='lbfgs',max_iter=2000,tol=1e-4,random_state=SEED))])
