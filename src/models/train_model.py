import pandas as pd
import os
import numpy as np
from stellargraph import StellarGraph
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML


def training(citeseer_content_feats, citeseer_cites):
    '''
    Trains a StellarGraph GCN with parameters
    '''
    
    
    graph = StellarGraph({"paper": citeseer_content_feats}, {"cites": citeseer_cites})
    dataset = sg.datasets.CiteSeer()
    graph, nodes = dataset.load()
    train_subjects, test_subjects = model_selection.train_test_split(nodes, 
                                                                     train_size = 1812, 
                                                                     test_size=None, 
                                                                     stratify=nodes
                                                                    )
    
    val_subjects, test_subjects = model_selection.train_test_split(test_subjects, 
                                                                   train_size = 500, 
                                                                   test_size=None, 
                                                                   stratify=test_subjects
                                                                  )
    
    target_encoding = preprocessing.LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    # create a FullBatchNodeGenerator object, as GCN is a full batch model.
    generator = FullBatchNodeGenerator(graph, method = "gcn")
    train_gen = generator.flow(train_subjects.index, train_targets) # produce object that can be used to train model
    
    # gcn with 2 layers 32 units each, softmax activations and 0.5 dropout
    gcn = GCN(layer_sizes = [32, 32], activations = ["softmax", "softmax"], generator=generator, dropout=0.5)
    x_inp, x_out = gcn.in_out_tensors()
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)
    
    model = Model(inputs = x_inp, outputs=predictions)
    model.compile(optimizer = optimizers.Adam(lr=0.01),
              loss = losses.categorical_crossentropy,
              metrics = ["acc"])
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    # stops early if convergence
    es_callback = EarlyStopping(monitor = "val_acc", patience=50, restore_best_weights=True)
    
    history = model.fit(
    train_gen,
    epochs = 200,
    validation_data = val_gen,
    verbose = 2,
    shuffle = False,
    callbacks = [es_callback])
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    # access when calling by index
    return model, test_gen, generator, nodes, target_encoding
    