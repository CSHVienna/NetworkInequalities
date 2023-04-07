########################################################
# Dependencies
########################################################

import numpy as np 
import networkx as nx
from collections import Counter
from sklearn import linear_model
from sklearn.preprocessing import normalize

from tmp.libs.handlers import utils

########################################################
# Constants
########################################################

SAMPLE_RANDOM = 'random'

LOCAL_PRIOR = 'class'

RELATIONAL_NBC = 'nBC'
RELATIONAL_LINK = 'LINK'

INFERENCE_RELAXATION = 'relaxation'

EPSILON = 1e-10 
LR_MAX_ITER = 1e4

################################################################################################################
# Local Model
################################################################################################################

class Local(object):
  def __init__(self, sample, kind):
    self.sample = sample  # sample (seeds info)
    self.kind = kind    # local model name
    self.prior = None   # priors
    self.classes = None # class names
    self.nclasses = 0
    
  def learn(self):
    if self.kind == LOCAL_PRIOR:
      self.learn_prior_class()
    else:
      utils.error(f"Prior.learn | collective.py | {self.kind} not implemented") 
    
  def learn_prior_class(self):
    counts = Counter(self.sample.Aseeds)
    self.classes, values = zip(*counts.most_common())
    self.nclasses = len(self.classes)
    values = np.array(values)
    self.prior = values / sum(values)

################################################################################################################
# Relational Model
################################################################################################################
    
class Relational(object):
  def __init__(self, sample, local, kind):
    self.sample = sample
    self.local = local
    self.kind = kind
    self.model = None
  
  ##########################
  def predict(self):
    if self.kind == RELATIONAL_NBC:
      return self.predict_nBC()
    elif self.kind == RELATIONAL_LINK:
      return self.predict_LINK()
    else:
      utils.error(f"Relational.predict | collective.py | {self.kind} not implemented")  
   
  def predict_nBC(self):
    return 
  
  def predict_LINK(self):
    return self.model.predict_proba(self.sample.Mtest) 
  
  ##########################
  def fit(self):
    if self.kind == RELATIONAL_NBC:
      self.fit_nBC()
    elif self.kind == RELATIONAL_LINK:
      self.fit_LINK()
    else:
      utils.error(f"Relational.fit | collective.py | {self.kind} not implemented")  

  def fit_nBC(self):
    # mixing matrix (normalized row-wise)
    self.model = np.zeros(shape=(self.local.nclasses,self.local.nclasses))
    for s, cs in enumerate(self.local.classes):
      inds = np.where(self.sample.Aseeds==cs)[0]
      for t, ct in enumerate(self.local.classes):
        indt = np.where(self.sample.Aseeds==ct)[0]
        self.model[s,t] = self.sample.Mseeds[inds,:][:,indt].sum()
    self.model += EPSILON #smooth to avoid probabilities of 0
    self.model = normalize(self.model, axis=1, norm='l1')
    
  def fit_LINK(self):
    self.model = linear_model.LogisticRegression(penalty='l2', C=10e20, solver='lbfgs', max_iter=LR_MAX_ITER)
    self.model.fit(self.sample.Mseeds, np.ravel(self.sample.Aseeds))
  
################################################################################################################
# Inference algorithm
################################################################################################################

class Inference(object):
  def __init__(self, relational, kind):
    self.relational = relational
    self.kind = kind
    
    n = self.relational.sample.get_number_of_nodes_in_test()
    p = self.relational.local.nclasses
    self.ci = np.ones(shape=(n,p))  # posterior prob
    self.xi = np.zeros(shape=(n,1))  # predicted class label
    
  def predict(self):
    if self.kind == INFERENCE_RELAXATION:
      self.predict_by_relaxation()
    else:
      utils.error(f"Inference.Inference | collective.py | {self.kind} not implemented")   
  
  def predict_by_relaxation(self):
    '''
    Predicts the new posterior and class label of nodes using relaxation labeling.
    Values are store per node as node attributes ci and xi respectively
    '''
    
    # hyper-params
    T = np.arange(0,99,1)
    k = 1.0
    alpha = 0.99
    
    # 1. Initialize with prior (step 1)
    self.ci *= self.relational.local.prior
    self.xi = np.random.choice(a=self.relational.local.classes, p=self.relational.local.prior, size=self.ci.shape[0])
    
    # 2. Estimate xi by applying the relational model T times (steps 2, ..., 100)
    beta = k
    for t in T:
      beta *= alpha
      self.ci = self._update_ci(beta) # new pior
      self.xi = self._update_xi()     # new class label
  
  def _update_ci(self, beta):
    '''
    Computes the new posterior probability for each node
    '''
    _prev = (1-beta) * self.ci
    _new = beta * self.relational.predict()
    return _new + _prev 
  
  def _update_xi(self):
    '''
    Computes the new class label xi for each node
    returns xi 
    '''
    return np.array([np.random.choice(a=self.relational.local.classes, p=ci) if ci[0]<=0.4 or ci[0]>=0.6 else self.relational.local.classes[ci.argmax()] for ci in self.ci])
  
################################################################################################################
# Sampling
################################################################################################################

class Sample(object):
  def __init__(self, nodes, M, A, percentage_nodes, method):
    self.nodes = nodes
    self.M = M
    self.A = A
    self.percentage_nodes = percentage_nodes
    self.size = int(round(self.percentage_nodes * len(self.nodes)))
    self.method = method
    
    self.seeds = None # known nodes (percentage of nodes in G)
    self.Mseeds = None # adjacency matrix from sample
    self.Aseeds = None # vector of seed class labels
    
    self.test = None # unknown nodes (percentage of nodes in G)
    self.Mtest = None # adjacency matrix from remaining nodes (G-Gseeds)
    self.Atest = None # vector of true class labels from remaining nodes (for evaluation purposes)
    
  def extract_sample_randomly(self):
    self.seeds = np.random.choice(a=self.nodes, size=self.size, replace=False)
    self.Mseeds = self.M[self.seeds,:] #[:,self.seeds]
    self.Aseeds = self.A[self.seeds]
    
    self.test = np.setdiff1d(self.nodes, self.seeds)
    self.Mtest = self.M[self.test,:] #[:,self.test]
    self.Atest = self.A[self.test]
    
  def extract_sample(self):
    if self.method == SAMPLE_RANDOM:
      self.extract_sample_randomly()
    else:
      utils.error(f"Sample.get_sample | collective.py | {self.method} not implemented")

  def get_number_of_nodes_in_test(self):
    return self.test.shape[0]

################################################################################################################
# Collective Classification
################################################################################################################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class CollectiveClassitication(object):
  prior = None
  relational = None
  inference = None
    
  def __init__(self, G):
    self.G = G # networkx undigraph
    self.M = None # adjacency matrix
    self.A = None # node class labels vector
    self.nodes = None # list of node ids (matching ids in M and A)
    self._nodes = None # lost of nodes (names from G)
    
    self._nodes = list(self.G.nodes())
    self.M = nx.to_scipy_sparse_array(G=self.G, nodelist=self._nodes)
    self.nodes = np.arange(len(self._nodes))
    self.A = np.array([G.nodes[n][G.graph['class']] for n in self._nodes])
    
    self.sample = None
    self.local = None
    self.relational = None
    self.inference = None

  def train_and_test(self, sample_size=0.1, sample_method='random', local_name='class', relational_name='nBC', inference_name='relaxation'):
    # sampling
    self.sample = Sample(self.nodes, self.M, self.A, sample_size, sample_method)
    self.sample.extract_sample()
    
    # priors
    self.local = Local(self.sample, local_name)
    self.local.learn()
    
    # relational
    self.relational = Relational(self.sample, self.local, relational_name)
    self.relational.fit()
    
    # inference
    self.inference = Inference(self.relational, inference_name)
    self.inference.predict()
    
  
  def evaluation(self, verbose=True):
    y_true = self.sample.Atest
    y_pred = self.inference.xi
    y_score = self.inference.ci
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(y_true)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y_true_arr = onehot_encoder.fit_transform(integer_encoded)
    
    cmn = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=self.local.classes, normalize='true')
    cm = confusion_matrix(y_true, y_pred, labels=self.local.classes)
    accuracy = accuracy_score(y_true, y_pred)
    average = 'weighted'
    precision = precision_score(y_true, y_pred, labels=self.local.classes, average=average)
    recall = recall_score(y_true, y_pred, labels=self.local.classes, average=average)
    f1 = f1_score(y_true, y_pred, labels=self.local.classes, average=average)
    rocauc = roc_auc_score(y_true_arr, y_score, labels=self.local.classes, average=average)
    
    if verbose:
      utils.info("================")
      utils.info("Evaluation:")
      utils.info("================")
      utils.info(f"- accuracy: {accuracy}")
      utils.info(f"- precision: {precision}")
      utils.info(f"- recall: {recall}")
      utils.info(f"- f1: {f1}")
      utils.info(f"- rocauc: {rocauc}")
      utils.info(f"Confusion:\n{cm}")
      utils.info(f"Confusion (true-norm)\n{cmn}")
    
    return {'accuracy':accuracy, 'precision':precision, 'recall':recall, 'f1':f1, 'rocauc':rocauc}