import argparse
import datetime
import os

import numpy as np
from mindspore import context

import train
from load_data import get_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr1', type=list, default=5e-5)
    parser.add_argument('--lr2', type=list, default=5e-5)
    parser.add_argument('--dataset', type=str, default='xmedia')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--k', type=float, default=5)
    parser.add_argument('--batchSize', type=int, default=64)

    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--commonSubspaceDim', type=int, default=200)
    parser.add_argument('--margin1', type=float, default=1)
    parser.add_argument('--margin2', type=float, default=0.5)

    parser.add_argument('--device', type=str, default='CPU')

    config = parser.parse_args()

    dataset = config.dataset
    epochs = config.epochs
    batchSize = config.batchSize
    device = config.device

    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)

    print('...Data loading is beginning...')
    dataLoader, inputData = get_loader(config.dataset, config.batchSize)
    print('...Data loading is completed...')

    print('-' * 50)
    print(f'config: dataset={dataset}, epochs={epochs}, batchSize={batchSize}')
    print('-' * 50)

    datetime_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = f'./result_new/{dataset}/epochs_{epochs}_batchSize_{batchSize}_{datetime_str}'

    os.makedirs(result_dir, exist_ok=True)

    print('...Training is beginning...')
    resultDict = train.trainModel(inputData=inputData, dataLoaders=dataLoader, config=config, result_dir=result_dir)
    print('...Training is completed...')

    for key, value in resultDict.items():
        np.save(f'{result_dir}/{key}.npy', value)

    print('...The results were saved successfully...')
