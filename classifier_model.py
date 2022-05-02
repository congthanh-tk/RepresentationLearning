import os
import json
import sys

import numpy
import torch
from sklearn.model_selection import train_test_split

from representation_learning_api import RepresentationLearningModel

if __name__ == '__main__':
    numpy.random.rand(1000)
    torch.manual_seed(1000)

    lambda1 = 0.5
    lambda2 = 0.001
    num_layers = 1
    output_dir = 'results_test'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = open(f"{output_dir}/result.tsv", 'w')

    
    X = numpy.array(features)
    Y = numpy.array(targets)

    print('Dataset', X.shape, Y.shape, numpy.sum(Y), sep='\t', file=sys.stderr)
    print('=' * 100, file=sys.stderr, flush=True)
    for _ in range(30):
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
        print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape, sep='\t', file=sys.stderr, flush=True)
        model = RepresentationLearningModel(
            lambda1=lambda1, lambda2=lambda2, batch_size=128, print=True, max_patience=5, balance=True,
            num_layers=num_layers
        )

        model.train(train_X, train_Y)
        results = model.evaluate(test_X, test_Y)

        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t', flush=True,
              file=output_file)

        print(results['accuracy'], results['precision'], results['recall'], results['f1'], sep='\t',
              file=sys.stderr, flush=True, end=('\n' + '=' * 100 + '\n'))

    output_file.close()
    
    pass
