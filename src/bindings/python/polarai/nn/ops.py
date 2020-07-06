from polarai import direct

from polarai.nn.globals import *

def init_polar():
  _global_context = direct.Context()
  _global_graph = direct.Graph(_global_context, "global_graph")

def add(a, b):
  pass

def mul(a, b):
  pass

def div(a, b):
  pass

def matmul(left, right):
  pass
