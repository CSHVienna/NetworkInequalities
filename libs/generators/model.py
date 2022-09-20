################################################################
# Dependencies
################################################################

import networkx as nx

from libs.generators import PAH as mPAH
from libs.generators import DH as mDH
from libs.generators import DPA as mDPA
from libs.generators import DPAH as mDPAH
from libs.generators import PATCH as mPATCH
from libs.generators import MAH


################################################################
# Generators' contructors
################################################################

def PAH(N=100, m=2, fm=0.1, h_MM=0.5, h_mm=0.5):
  G = mPAH.create(N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm)
  set_metadata(G, N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm)
  return G

def DH(N=100, fm=0.1, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=0.2, plo_m=0.2):
  G = mDH.create(N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m)
  set_metadata(G, N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m)
  return G

def DPA(N=100, fm=0.1, d=0.1, plo_M=0.2, plo_m=0.2):
  G = mDPA.create(N=N, fm=fm, d=d, plo_M=plo_M, plo_m=plo_m)
  set_metadata(G, N=N, fm=fm, d=d, plo_M=plo_M, plo_m=plo_m)
  return G

def DPAH(N=100, fm=0.1, d=0.1, h_MM=0.5, h_mm=0.5, plo_M=0.2, plo_m=0.2):
  G = mDPAH.create(N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m)
  set_metadata(G, N=N, fm=fm, d=d, h_MM=h_MM, h_mm=h_mm, plo_M=plo_M, plo_m=plo_m)
  return G

def PATCH(N=100, m=2, fm=0.1, h_MM=0.5, h_mm=0.5, tc=0.1):
  G = mPATCH.create(N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, triadic_closure=tc)
  set_metadata(G, N=N, m=m, fm=fm, h_MM=h_MM, h_mm=h_mm, tc=tc)
  return G

def MABAH():
  return

################################################################
# Functions
################################################################

def set_metadata(G, **kwargs):
  for k,v in kwargs.items():
    G.graph[k] = v
  