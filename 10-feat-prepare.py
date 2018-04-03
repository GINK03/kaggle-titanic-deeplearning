import os

import sys

from pathlib import Path

import csv

import json

import pickle
# make feat_index.json
if '--step1' in sys.argv:
  HOME = os.environ['HOME']

  objs = []
  for type in ['train', 'test']:
    fp = open(f'{HOME}/.kaggle/competitions/titanic/{type}.csv')
    csvfp = csv.reader(fp)

    head = next(csvfp)

    for line in csvfp:
      obj = dict( zip(head, line) )

      del obj['Name']
      del obj['Ticket']
      del obj['Cabin']

      obj2 = {}
      for k, v in obj.items():
        try:
          obj2[k] = float(v)
        except Exception as ex:
          obj2[k] = v if v != '' else None
      #print(obj2)
      objs.append( obj2 )

  print( json.dumps( objs, indent=2 ) )

  # feat uniq count
  feats = set()
  for obj in objs:
    for key, val in obj.items():
      if val is not None and key in ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        feats.add( key ) 
      else:
        feat = f'{key}:{val}'
        feats.add( feat )

  print( feats )

  feat_index = {feat:index for index, feat in enumerate(feats) }
  print( feat_index )

  json.dump( feat_index, fp=open('feat_index.json', 'w') )

# make dense
if '--step2' in sys.argv:
  
  feat_index = json.load( fp=open('feat_index.json') )
  HOME = os.environ['HOME']

  for type in ['train', 'test']:
    fp = open(f'{HOME}/.kaggle/competitions/titanic/{type}.csv')
    print(fp)
    csvfp = csv.reader(fp)

    head = next(csvfp)

    objs = []
    for line in csvfp:
      obj = dict( zip(head, line) )

      del obj['Name']
      del obj['Ticket']
      del obj['Cabin']

      obj2 = {}
      for k, v in obj.items():
        try:
          obj2[k] = float(v)
        except Exception as ex:
          obj2[k] = v if v != '' else None
      #print(obj2)
      objs.append( obj2 )
    
    feat_index = json.load( fp=open('feat_index.json') )
    Xs, ys = [], []
    for obj in objs:
      x = [0.0]*len(feat_index)
      y = 0
      for key, val in obj.items():
        # print(key)
        if key == 'Survived':
          y = val 
          continue
        if val is not None and key in ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
          x[ feat_index[key] ] = val 
        else:
          feat = f'{key}:{val}'
          x[ feat_index[feat] ] = 1.0

      print(type, y, x)
      Xs.append(x); ys.append(y)
    pickle.dump( (Xs, ys), open(f'{type}.pkl', 'wb') )  
