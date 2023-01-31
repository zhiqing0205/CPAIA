import os
import time
import numpy as np
from mindspore import nn, Tensor, ops, load_checkpoint, save_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindspore.common.initializer import Zero
import model
import scipy.spatial


def cal_MAP(query, db, label, k=0, dist_method='L2'):
    dist = ''
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(query, db, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(query, db, 'cosine')
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    return np.mean(res)


def trainModel(inputData, dataLoaders, config, result_dir):
    k = config.k
    margin1 = config.margin1
    margin2 = config.margin2
    epochs = config.epochs
    commonSubspaceDim = config.commonSubspaceDim

    mergeNet = model.MergeNet(imgInputDim=inputData['img_dim'], textInputDim=inputData['text_dim'],
                              commonSubspaceDim=commonSubspaceDim, semanticSubspaceDim=inputData['num_class'])
    modaliyDiscriminator = model.ModaliyDiscriminator(inputDim=config.commonSubspaceDim)
    representationClassifier = model.RepresentationClassifier()
    optimizer1 = nn.Adam(mergeNet.trainable_params(), learning_rate=config.lr1, beta1=config.beta1, beta2=config.beta2)
    optimizer2 = nn.Adam(modaliyDiscriminator.trainable_params(), learning_rate=config.lr2, beta1=config.beta1,
                         beta2=config.beta2)
    optimizer3 = nn.Adam(representationClassifier.trainable_params(), learning_rate=config.lr1, beta1=config.beta1, beta2=config.beta2)

    imgTrain = Tensor(inputData["img_train"], mstype.float32)
    textTrain = Tensor(inputData["text_train"], mstype.float32)
    labelTrain = Tensor(inputData["label_train"], mstype.float32)
    label_Train = labelTrain.asnumpy()

    imgCommonRepresentation, textCommonRepresentation,_,_,_,_,_,_ = mergeNet(imgTrain, textTrain)

    imgCenter = Tensor(shape=(inputData["num_class"], commonSubspaceDim), dtype=mstype.float32, init=Zero())
    textCenter = Tensor(shape=(inputData["num_class"], commonSubspaceDim), dtype=mstype.float32, init=Zero())

    visited = []
    for label in label_Train:
        labelIndice = np.where(label > 0)[0][0]
        if labelIndice in visited:
            continue
        sampleIndiceMask = np.matmul(label_Train, np.transpose(label)) > 0
        sampleIndice = np.where(sampleIndiceMask)[0]
        imgCenter[Tensor(np.where(label > 0)[0], mstype.int32)] = ops.reduce_mean(
            imgCommonRepresentation[Tensor(sampleIndice)], 0)
        textCenter[Tensor(np.where(label > 0)[0], mstype.int32)] = ops.reduce_mean(
            textCommonRepresentation[Tensor(sampleIndice)], 0)
        visited.append(labelIndice)

    retrievalLoss = model.RetrievalLoss(margin1, margin2, imgCenter, textCenter)
    semanticLoss = model.SemanticLoss()
    modalityLoss = model.ModalityLoss()
    representationClassificationLoss = model.RepresentationClassificationLoss()

    #
    withLossCellR = model.WithLossCellR(mergeNet, retrievalLoss)
    withLossCellS = model.WithLossCellS(mergeNet, semanticLoss)
    withLossCellG = model.WithLossCellG(mergeNet, modaliyDiscriminator, modalityLoss)
    withLossCellD = model.WithLossCellD(mergeNet, modaliyDiscriminator, modalityLoss)
    withLossCellC = model.WithLossCellC(mergeNet, representationClassifier, representationClassificationLoss)

    #
    myTrainOneStepCellForR = nn.TrainOneStepCell(withLossCellR, optimizer1)
    myTrainOneStepCellForS = nn.TrainOneStepCell(withLossCellS, optimizer1)
    myTrainOneStepCellForG = nn.TrainOneStepCell(withLossCellG, optimizer1)
    myTrainOneStepCellForD = nn.TrainOneStepCell(withLossCellD, optimizer2)
    myTrainOneStepCellForC = nn.TrainOneStepCell(withLossCellC, optimizer3)

    cpaia = model.CPAIA(myTrainOneStepCellForR, myTrainOneStepCellForS, myTrainOneStepCellForG)

    ###################################################################################################################
    # 记录并保存数据
    bestResult = 0.0
    bestRecord = []

    trReLosses = []
    trSeLosses = []
    trEmLosses = []
    trGeLosses = []
    trMoLosses = []
    trClLosses = []

    test_mAP = []
    ###################################################################################################################

    since = time.time()

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 30)

        for phase in ['train', 'valid']:

            meanReLoss = []
            meanSeLoss = []
            meanEmLoss = []
            meanGeLoss = []
            meanMoLoss = []
            meanClLoss = []

            for i, (imgs, texts, labels) in enumerate(dataLoaders[phase]):
                batch = i + 1

                if (imgs != imgs).sum() > 1 or (texts != texts).sum() > 1:
                    print("Data contains Nan.")
                    continue

                reLoss, seLoss, geLoss = cpaia(imgs, texts, labels)

                meanReLoss.append(reLoss.asnumpy())
                meanSeLoss.append(seLoss.asnumpy())
                meanEmLoss.append(reLoss.asnumpy() + seLoss.asnumpy())
                meanGeLoss.append(geLoss.abs().asnumpy())

                if batch % k == 0:
                    moLoss = myTrainOneStepCellForD(imgs, texts)
                    clLoss = myTrainOneStepCellForC(imgs, texts)

                    meanMoLoss.append(moLoss.asnumpy())
                    meanClLoss.append(clLoss.asnumpy())

            trReLosses.append(np.mean(meanReLoss))
            trSeLosses.append(np.mean(meanSeLoss))
            trEmLosses.append(np.mean(meanEmLoss))
            trGeLosses.append(np.mean(meanGeLoss))
            trMoLosses.append(np.mean(meanMoLoss))
            trClLosses.append(np.mean(meanClLoss))


            print('{} reloss: {:.4f}  seloss: {:.4f}  emdloss: {:.4f}  geloss: {:.4f}  moloss: {:.4f}'.format(
                phase, np.mean(meanReLoss), np.mean(meanSeLoss), np.mean(meanEmLoss), np.mean(meanGeLoss),
                np.mean(meanMoLoss)))

            if phase == "valid":
                imgsList, textsList, labelsList = [], [], []
                for imgs, texts, labels in dataLoaders['test']:
                    imgsCommonRe, textsCommonRe, _, _,_,_,_,_ = mergeNet(imgs, texts)

                    imgsList.append(imgsCommonRe.asnumpy())
                    textsList.append(textsCommonRe.asnumpy())
                    labelsList.append(labels.asnumpy())

                imgsList = np.concatenate(imgsList)
                textsList = np.concatenate(textsList)
                labelsList = np.concatenate(labelsList).argmax(1)
                img2text = cal_MAP(imgsList, textsList, labelsList)
                text2img = cal_MAP(textsList, imgsList, labelsList)
                test_mAP.append(((img2text + text2img) / 2.))

                print("test_mAP: {:.4f}  (Img2Txt: {:.4f}  Txt2Img: {:.4f})".format((img2text + text2img) / 2, img2text,
                                                                                    text2img))

                if (img2text + text2img) / 2. > bestResult:
                    bestResult = (img2text + text2img) / 2.
                    bestRecord = [img2text, text2img]
                    save_checkpoint(mergeNet, os.path.join(result_dir, 'best_mergeNet.ckpt'))
                    # save_checkpoint(mergeNet, "bestMergeNet.ckpt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best result: {:.4f}  Best record: ({:.4f}  {:.4f})'.format(bestResult, *bestRecord))

    resultDict = {}
    resultDict.setdefault("trReLosses", trReLosses)
    resultDict.setdefault("trSeLosses", trSeLosses)
    resultDict.setdefault("trEmLosses", trEmLosses)
    resultDict.setdefault("trGeLosses", trGeLosses)
    resultDict.setdefault("trMoLosses", trMoLosses)

    resultDict.setdefault("test_mAP", test_mAP)
    resultDict.setdefault("bestResult", bestResult)

    return resultDict
