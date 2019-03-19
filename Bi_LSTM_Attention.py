#coding: utf-8
import os
import io
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score
warnings.filterwarnings("ignore")
class TrainingConfig(object):
    epoches = 6
    evaluateEvery = 50
    checkpointEvery = 100
    domytestEvery = 500
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    hiddenSizes = [256,128]

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    sequenceLength = 200
    batchSize = 128
    dataSource = "labeledTrain.csv"
    stopWordSource = "english"
    numClasses = 2
    rate = 0.8
    training = TrainingConfig()
    model = ModelConfig()


config = Config()


class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._wordToIndex = {}
        self._indexToWord = {}

    def _readStopWord(self, stopWordPath):

        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词以列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
            """首先读取停用词"""


    def _readData(self,filePath):
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist()
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]
        return reviews,labels
    """读取数据"""

    def _genVocabulary(self, reviews):


        allWords = [word for review in reviews for word in review]#训练集所有词汇含重复

        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(subWords)  # 统计词频,且无重复 {'like':123,'love':253,....}
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)#按照词频排序，且为逆序

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5] #只保留了词

        vocab, wordEmbedding = self._getWordEmbedding(words) #调用下面的函数
        self.wordEmbedding = wordEmbedding

        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))

    def _getWordEmbedding(self, words):
        """被上一函数调用，功能为得到去掉了停用词后的词表和词向量表"""
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)#读取训练好了词训练表
        vocab = []
        wordEmbedding = []
        vocab.append("pad")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize)) #填充的 index为0的词向量
        wordEmbedding.append(np.random.randn(self._embeddingSize))#未知词的词向量

        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
        return vocab, np.array(wordEmbedding) #返回词表和词向量表


    def _genTrainEvalData(self, x, y, rate):
        """将数据集中所有评论用index表示，用0进行填充"""
        """训练数据并没有去掉低频词和停用词"""
        reviews = []
        labels = []

        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)#调用下面的函数, x[i]表示一个评论
            reviews.append(reviewVec) #得到一个评论的index表示，然后继续循环得到所有的

            labels.append([y[i]])
        trainIndex = int(len(x) * rate) #划分训练集和测试集

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _reviewProcess(self,review,sequenceLength,wordToIndex):
        """将数据集中每条评论用index表示，用0进行填充"""
        reviewVec = np.zeros((sequenceLength)) #用0进行填充
        sequenceLen = sequenceLength
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]#对评论中每一个词

            else:
                reviewVec[i] = wordToIndex["UNK"]   #去掉了停用词和低频词，可能不会出现在词表中

        return reviewVec



    def dataGen(self):
        """运行所有函数的函数"""
        # 初始化停用词
        self._readStopWord(self._stopWordSource)

        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels

    def dosometest(self, filePath):
        df = pd.read_csv(filePath)
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]
        test_reviews = []

        for i in range(len(reviews)):
            reviewVec = self._reviewProcess(reviews[i], self._sequenceLength, self._wordToIndex)
            test_reviews.append(reviewVec)

        test_reviews = np.asarray(test_reviews, dtype="int64")
        for i in range(10):
            start = i*2500
            end = start + 2500
            review = test_reviews[start:end]
            yield review



    def gettestid(self,filePath):
        df = pd.read_csv(filePath)
        id = df['id']
        return id

data = Dataset(config)
data.dataGen()
"""以上生成了训练数据和测试数据"""




def nextBatch(x, y, batchSize):
    #np.random.seed(10)

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


class BiLSTMAttention(object):
    def __init__(self,config,wordEmbedding):
        self.inputX = tf.placeholder(tf.int32,[None,config.sequenceLength],name='inputX')
        self.inputY = tf.placeholder(tf.float32,[None,1],name='inputY')
        self.dropoutKeepProb = tf.placeholder(tf.float32,name='dropoutKeepProb')

        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.W = tf.Variable(tf.cast(wordEmbedding,dtype=tf.float32,name="word2vec"),name="W")
            self.embeddedWords = tf.nn.embedding_lookup(self.W,self.inputX)

        with tf.name_scope("Bi-LSTM"):
            """定义双层双向LSTM"""
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(#前向
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(#后向
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx))
                    # outputs_是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    self.embeddedWords = tf.concat(outputs_, 2)# 把outputs中的fw和bw的结果拼接为[batch_size, time_step, hidden_size * 2]


        outputs = tf.split(self.embeddedWords,2,-1) # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出 维度都是[batch_size, max_time, hidden_size]
        with tf.name_scope("Attention"):
            H =outputs[0] + outputs[1]  #在Bi-LSTM+Attention论文中，将前向和后向的输出相加
            output =self.attention(H) #Attention
            outputSize = config.model.hiddenSizes[-1]





        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, 1],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(output, outputW, outputB, name="predictions")
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")

        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = config.model.hiddenSizes[-1]

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)  #维度不变

        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]

        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1])) #newM = [batch_size*time_step, 1]，每一个时间步的输出由向量转换成一个数字


        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, config.sequenceLength])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, config.sequenceLength, 1])) #tf.transpose:三维矩阵的转置

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.squeeze(r)  #移除大小为1的维度

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutKeepProb)

        return output


def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)
# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
MODEL_SAVE_PATH = "model_attention"
MODEL_NAME = "model.ckpt"
# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.95  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        lstm = BiLSTMAttention(config, wordEmbedding)

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(lstm.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        """
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summary"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", lstm.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
        """
        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            """ 加载模型 """




        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _,  step, loss, predictions, binaryPreds = sess.run(
                [trainOp, globalStep, lstm.loss, lstm.predictions, lstm.binaryPreds],
                feed_dict)


            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc,
                                                                                               auc, precision, recall))
            #trainSummaryWriter.add_summary(summary, step)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                lstm.inputX: batchX,
                lstm.inputY: batchY,
                lstm.dropoutKeepProb: 1.0
            }
            step, loss, predictions, binaryPreds = sess.run(
                [globalStep, lstm.loss, lstm.predictions, lstm.binaryPreds],
                feed_dict)

            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)

            #evalSummaryWriter.add_summary(summary, step)

            return loss, acc, auc, precision, recall


        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)
                        break

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accs),
                                                                                                       mean(aucs),
                                                                                                       mean(precisions),
                                                                                                       mean(recalls)))


                if currentStep % config.training.checkpointEvery == 0:
                    saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=globalStep)

                    """ 保存模型 """

                if currentStep % config.training.domytestEvery == 0:
                    pred = []

                    for batchtest in data.dosometest("test.csv"):
                        predict = sess.run(lstm.binaryPreds,
                                                    feed_dict={lstm.inputX: batchtest,
                                                               lstm.dropoutKeepProb: 1.0})
                        pred=np.append(pred,predict)




                    result = pd.DataFrame(
                        {'id': data.gettestid("test.csv").as_matrix(), 'sentiment': pred.astype(np.int32)})

                    result.to_csv("result.csv", index=False)
                    """保存结果"""

























