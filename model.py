from mindspore import nn, Tensor, ops
import numpy as np
from mindspore import dtype as mstype
import itertools


class ImgSubNet(nn.Cell):
    def __init__(self, inputDim=4096, commonSubspaceDim=200):
        super(ImgSubNet, self).__init__()
        midDim1 = 2048
        midDim2 = 1024
        self.fc1 = nn.Dense(inputDim, midDim1, bias_init="normal")
        self.fc2 = nn.Dense(midDim1, midDim2, bias_init="normal")
        self.fc3 = nn.Dense(midDim2, commonSubspaceDim, bias_init="normal")
        self.fc4 = nn.Dense(commonSubspaceDim, commonSubspaceDim, bias_init="normal")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out1 = self.relu(self.fc1(x))
        out2 = self.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        out4 = self.sigmoid(self.fc4(out3))
        shared = ops.mul(out3, out4)
        specific = ops.mul(out3, 1 - out4)
        return shared, specific


class TextSubNet(nn.Cell):
    def __init__(self, inputDim=3000, commonSubspaceDim=200):
        super(TextSubNet, self).__init__()
        midDim1 = 2048
        midDim2 = 1024
        self.fc1 = nn.Dense(inputDim, midDim1, bias_init="normal")
        self.fc2 = nn.Dense(midDim1, midDim2, bias_init="normal")
        self.fc3 = nn.Dense(midDim2, commonSubspaceDim, bias_init="normal")
        self.fc4 = nn.Dense(commonSubspaceDim, commonSubspaceDim, bias_init="normal")
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        out1 = self.relu(self.fc1(x))
        out2 = self.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        out4 = self.sigmoid(self.fc4(out3))
        shared = ops.mul(out3, out4)
        specific = ops.mul(out3, 1 - out4)
        return shared, specific


class MergeNet(nn.Cell):
    def __init__(self, imgInputDim=4096, textInputDim=3000, commonSubspaceDim=200, semanticSubspaceDim=20):
        super(MergeNet, self).__init__()
        self.imgSubNet = ImgSubNet(imgInputDim, commonSubspaceDim)
        self.textSubNet = TextSubNet(textInputDim, commonSubspaceDim)

        self.fc1 = nn.Dense(commonSubspaceDim, semanticSubspaceDim, bias_init="normal")
        self.fc2 = nn.Dense(commonSubspaceDim, 3, bias_init="normal")

        self.softmax = nn.Softmax(axis=1)
        self.norm = nn.Norm(axis=1, keep_dims=True)

    def construct(self, imgFeature, textFeature):
        imgSharedFeature, imgSpecificFeature = self.imgSubNet(imgFeature)
        textSharedFeature, textSpecificFeature = self.textSubNet(textFeature)

        imgNorm = self.norm(imgSharedFeature)
        textNorm = self.norm(textSharedFeature)
        imgCommonRepresentation = imgSharedFeature / imgNorm
        textCommonRepresentation = textSharedFeature / textNorm

        imgSemanticRepresentation = self.fc1(imgCommonRepresentation)
        textSemanticRepresentation = self.fc1(textCommonRepresentation)

        imgSharedeRepreClassifier = self.softmax(self.fc2(imgSharedFeature))
        textSharedeRepreClassifier = self.softmax(self.fc2(textSharedFeature))
        imgSpecificRepreClassifier = self.softmax(self.fc2(imgSpecificFeature))
        textSpecificRepreClassifier = self.softmax(self.fc2(textSpecificFeature))

        return imgCommonRepresentation, textCommonRepresentation, imgSemanticRepresentation, textSemanticRepresentation, imgSharedeRepreClassifier, textSharedeRepreClassifier, imgSpecificRepreClassifier, textSpecificRepreClassifier


class ModaliyDiscriminator(nn.Cell):
    def __init__(self, inputDim=200, outputDim=1):
        super(ModaliyDiscriminator, self).__init__()
        self.fc1 = nn.Dense(inputDim, 128, bias_init="normal")
        self.fc2 = nn.Dense(128, 64, bias_init="normal")
        self.fc3 = nn.Dense(64, outputDim, bias_init="normal")

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def construct(self, imgFeature, textFeature):
        imgFeature = self.relu(self.fc1(imgFeature))
        textFeature = self.relu(self.fc1(textFeature))

        imgFeature = self.relu(self.fc2(imgFeature))
        textFeature = self.relu(self.fc2(textFeature))

        imgModalPredict = self.sigmoid(self.fc3(imgFeature))
        textModalPredict = self.sigmoid(self.fc3(textFeature))

        return imgModalPredict, textModalPredict


class RetrievalLoss(nn.Cell):
    def __init__(self, margin1, margin2, imgCenter, textCenter):
        super(RetrievalLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.imgCenter = imgCenter
        self.textCenter = textCenter

        self.relu = nn.ReLU()
        self.softmax = ops.Softmax(axis=1)

    def construct(self, imgs, texts, labels):
        def calLabelSim(label1, label2):
            sim = ops.matmul(label1, label2.T)
            return sim

        def calPosCenter(i, posSampleIndice):
            attenMatrix1 = self.softmax(ops.matmul(imgs[i].reshape(1, -1), texts[Tensor(posSampleIndice)].T))
            attenMatrix2 = self.softmax(ops.matmul(texts[i].reshape(1, -1), imgs[Tensor(posSampleIndice)].T))
            i2tCenter = ops.matmul(attenMatrix1, texts[Tensor(posSampleIndice)]).squeeze()
            t2iCenter = ops.matmul(attenMatrix2, imgs[Tensor(posSampleIndice)]).squeeze()
            return i2tCenter.asnumpy(), t2iCenter.asnumpy()

        criterion = lambda x, y: ((x - y) ** 2).sum(1)
        cos = lambda x, y: ops.matmul(x, y.T) \
                           / ops.matmul(ops.sqrt((x ** 2).sum(1, keepdims=True)),
                                        ops.sqrt((y ** 2).sum(1, keepdims=True)).T).clip(xmin=1e-6, xmax=None) / 2.

        labels_ = labels.asnumpy()

        indiceList = []
        i2tCrossAttentionCenter = []
        t2iCrossAttentionCenter = []

        for i, label in enumerate(labels_):
            posLabelIndice = np.where(label > 0)[0][0]
            negLabelIndice = np.where(label == 0)[0]

            posSampleMask = np.matmul(labels_, np.transpose(label)) > 0
            posSampleIndice = np.where(posSampleMask)[0]

            i2tCenter, t2iCenter = calPosCenter(i, posSampleIndice)
            i2tCrossAttentionCenter.append(i2tCenter)
            t2iCrossAttentionCenter.append(t2iCenter)

            self.imgCenter[Tensor(np.where(label > 0)[0], mstype.int32)] = ops.reduce_mean(
                imgs[Tensor(posSampleIndice)], 0)
            self.textCenter[Tensor(np.where(label > 0)[0], mstype.int32)] = ops.reduce_mean(
                texts[Tensor(posSampleIndice)], 0)

            negLabelIndicePair = list(itertools.combinations(negLabelIndice, 2))  # 返回负类标签序列中所有的二元组
            tmp = [[i, j, neg[0], neg[1]] for i in posSampleIndice for j in negLabelIndice for neg in
                   negLabelIndicePair]
            indiceList += tmp

        indiceList = np.array(indiceList)

        i2tCrossAttentionCenter = Tensor(np.array(i2tCrossAttentionCenter))
        t2iCrossAttentionCenter = Tensor(np.array(t2iCrossAttentionCenter))

        imgTextLoss1 = criterion(imgs[Tensor(indiceList[:, 0])], i2tCrossAttentionCenter[Tensor(indiceList[:, 0])]) - \
                       criterion(imgs[Tensor(indiceList[:, 0])],
                                 self.textCenter[Tensor(indiceList[:, 1])]) + self.margin1
        imgTextLoss1 = self.relu(imgTextLoss1).mean()
        imgTextLoss2 = criterion(imgs[Tensor(indiceList[:, 0])], i2tCrossAttentionCenter[Tensor(indiceList[:, 0])]) - \
                       criterion(self.textCenter[Tensor(indiceList[:, 2])],
                                 self.textCenter[Tensor(indiceList[:, 3])]) + self.margin2
        imgTextLoss2 = self.relu(imgTextLoss2).mean()
        imgTextLoss = imgTextLoss1 + imgTextLoss2

        textImgLoss1 = criterion(texts[Tensor(indiceList[:, 0])], t2iCrossAttentionCenter[Tensor(indiceList[:, 0])]) - \
                       criterion(texts[Tensor(indiceList[:, 0])],
                                 self.imgCenter[Tensor(indiceList[:, 1])]) + self.margin1
        textImgLoss1 = self.relu(textImgLoss1).mean()
        textImgLoss2 = criterion(texts[Tensor(indiceList[:, 0])], t2iCrossAttentionCenter[Tensor(indiceList[:, 0])]) - \
                       criterion(self.imgCenter[Tensor(indiceList[:, 2])],
                                 self.imgCenter[Tensor(indiceList[:, 3])]) + self.margin2
        textImgLoss2 = self.relu(textImgLoss2).mean()
        textImgLoss = textImgLoss1 + textImgLoss2

        labelsSim = calLabelSim(labels.astype(mstype.float32), labels.astype(mstype.float32))
        fineDis = cos(texts, imgs)
        fineLoss = (ops.log(1 + ops.exp(fineDis)) - labelsSim * fineDis).mean()

        return imgTextLoss, textImgLoss, fineLoss


class SemanticLoss(nn.Cell):
    def __init__(self):
        super(SemanticLoss, self).__init__()

    def construct(self, imgs, texts, labels):
        loss = ops.sqrt(((imgs - labels) ** 2).sum(1)).mean() + ops.sqrt(((texts - labels) ** 2).sum(1)).mean()
        return loss


class ModalityLoss(nn.Cell):
    def __init__(self):
        super(ModalityLoss, self).__init__()
        self.bceLoss = nn.BCELoss(reduction='mean')
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()

    def construct(self, imgModalPredict, textModalPredict):
        one_matrix = self.ones((imgModalPredict.shape[0], 1), mstype.float32)
        zero_matirx = self.zeros((textModalPredict.shape[0], 1), mstype.float32)
        loss = self.bceLoss(imgModalPredict, one_matrix) + self.bceLoss(textModalPredict, zero_matirx)
        return loss


class WithLossCellR(nn.Cell):
    def __init__(self, mergeNet, retrievalLoss):
        super(WithLossCellR, self).__init__()
        self.mergeNet = mergeNet
        self.retrievalLoss = retrievalLoss

    def construct(self, imgs, texts, labels):
        imgCommonRepresentation, textCommonRepresentation, _, _, _, _, _, _ = self.mergeNet(imgs, texts)
        imgTextLoss, textImgLoss, fineLoss = self.retrievalLoss(imgCommonRepresentation, textCommonRepresentation,
                                                                labels)
        return imgTextLoss + textImgLoss + 0.5 * fineLoss


class WithLossCellS(nn.Cell):
    def __init__(self, mergeNet, semanticLoss):
        super(WithLossCellS, self).__init__()
        self.mergeNet = mergeNet
        self.semanticLoss = semanticLoss

    def construct(self, img, text, label):
        _, _, imgSemanticRepresentation, textSemanticRepresentation, _, _, _, _ = self.mergeNet(img, text)
        loss = self.semanticLoss(imgSemanticRepresentation, textSemanticRepresentation, label)
        return loss


class WithLossCellG(nn.Cell):
    def __init__(self, mergeNet, modalityDiscriminator, modalityLoss):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.mergeNet = mergeNet
        self.modalityDiscriminator = modalityDiscriminator
        self.modalityLoss = modalityLoss

    def construct(self, img, text):
        imgCommonRepresentation, textCommonRepresentation, _, _, _, _, _, _ = self.mergeNet(img, text)
        imgModalPredict, textModalPredict = self.modalityDiscriminator(imgCommonRepresentation,
                                                                       textCommonRepresentation)
        loss = self.modalityLoss(imgModalPredict, textModalPredict)
        return -loss


class WithLossCellD(nn.Cell):
    def __init__(self, mergeNet, modalityDiscriminator, modalityLoss):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.mergeNet = mergeNet
        self.modalityDiscriminator = modalityDiscriminator
        self.modalityLoss = modalityLoss

    def construct(self, img, text):
        imgCommonRepresentation, textCommonRepresentation, _, _, _, _, _, _ = self.mergeNet(img, text)
        imgModalPredict, textModalPredict = self.modalityDiscriminator(imgCommonRepresentation,
                                                                       textCommonRepresentation)
        loss = self.modalityLoss(imgModalPredict, textModalPredict)
        return loss


class RepresentationClassifier(nn.Cell):
    def __init__(self):
        super(RepresentationClassifier, self).__init__()
        self.fc = nn.Dense(3, 3, bias_init="normal")
        self.softmax = nn.Softmax(axis=1)
        # 三分类损失
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, representationFeature):
        representationPredict = self.softmax(self.fc(representationFeature))
        return representationPredict


class RepresentationClassificationLoss(nn.Cell):
    def __init__(self):
        super(RepresentationClassificationLoss, self).__init__()
        #
        self.bceloss = nn.BCELoss(reduction='mean')

    def construct(self, representationPredict, representationLabel):
        loss = self.bceloss(representationPredict, representationLabel)
        return loss


class WithLossCellC(nn.Cell):
    def __init__(self, mergeNet, representationClassifier, representationClassificationLoss):
        super(WithLossCellC, self).__init__(auto_prefix=True)
        self.mergeNet = mergeNet
        self.representationClassifier = representationClassifier
        self.representationClassificationLoss = representationClassificationLoss

    def construct(self, img, text):
        _, _, _, _, imgSharedeRepreClassifier, textSharedeRepreClassifier, imgSpecificRepreClassifier, textSpecificRepreClassifier = self.mergeNet(
            img, text)

        imgSharedeRepreClassifierLoss = self.representationClassificationLoss(self.representationClassifier(imgSharedeRepreClassifier), Tensor([[0, 1, 0] for _ in range(imgSharedeRepreClassifier.shape[0])], mstype.float32))
        textSharedeRepreClassifierLoss = self.representationClassificationLoss(self.representationClassifier(textSharedeRepreClassifier), Tensor([[0, 1, 0] for _ in range(textSharedeRepreClassifier.shape[0])] , mstype.float32))
        imgSpecificRepreClassifierLoss = self.representationClassificationLoss(self.representationClassifier(imgSpecificRepreClassifier), Tensor([[1, 0, 0] for _ in range(imgSpecificRepreClassifier.shape[0])] , mstype.float32))
        textSpecificRepreClassifierLoss = self.representationClassificationLoss(self.representationClassifier(textSpecificRepreClassifier), Tensor([[0, 0, 1] for _ in range(textSpecificRepreClassifier.shape[0])] , mstype.float32))
        loss = imgSharedeRepreClassifierLoss + textSharedeRepreClassifierLoss + imgSpecificRepreClassifierLoss + textSpecificRepreClassifierLoss
        return loss


class CPAIA(nn.Cell):
    def __init__(self, myTrainOneStepCellForR, myTrainOneStepCellForS, myTrainOneStepCellForG):
        super(CPAIA, self).__init__()
        self.myTrainOneStepCellForR = myTrainOneStepCellForR
        self.myTrainOneStepCellForS = myTrainOneStepCellForS
        self.myTrainOneStepCellForG = myTrainOneStepCellForG

    def construct(self, imgs, texts, labels):
        reloss = self.myTrainOneStepCellForR(imgs, texts, labels)
        seloss = self.myTrainOneStepCellForS(imgs, texts, labels)
        geloss = self.myTrainOneStepCellForG(imgs, texts)

        return reloss, seloss, geloss
