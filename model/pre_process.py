#导入库  正则 分词
import re,jieba,random,time
import numpy as np
import torch
from sklearn.model_selection import train_test_split  # 数据集的划分


#定义语料的类
class Corpus:
    #初始化
    def __init__(self, filePath, maxSentenceWordsNum=-1, id2word=None, word2id=None, wordNum=None, tfidf=False, QIDF=None, AIDF=None, testSize=0.2, isCharVec=False):
        self.id2word, self.word2id, self.wordNum = id2word, word2id, wordNum
        #遍历所有内容
        with open(filePath,'r',encoding='utf8') as f:
            txt = self._purify(f.readlines())
            data = [i.split('\t') for i in txt]
            #判断是否是字符，使用jieba进行分词
            if isCharVec:
                data = [[[c for c in i[0]], [c for c in i[1]]] for i in data]
            else:
                data = [[jieba.lcut(i[0]), jieba.lcut(i[1])] for i in data]
            data = [i for i in data if (len(i[0])<maxSentenceWordsNum and len(i[1])<maxSentenceWordsNum) or maxSentenceWordsNum==-1]
        self.chatDataWord = data
        self._word_id_map(data)

        try:
            chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
        except:
            chatDataId = [[[self.word2id[w] for w in qa[0] if w in self.id2word],[self.word2id[w] for w in qa[1] if w in self.id2word]] for qa in self.chatDataWord]
        
        self._QALens(chatDataId)
        self.maxSentLen = max(maxSentenceWordsNum, self.AMaxLen)
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
        self.totalSampleNum = len(data)
        #定义如果使用tfidf 手动实现
        if tfidf:
            self.QIDF = QIDF if QIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[0]) for qa in chatDataId])+1)) for i in range(self.wordNum)], dtype='float32')
            self.AIDF = AIDF if AIDF is not None else np.array([np.log(self.totalSampleNum/(sum([(i in qa[1]) for qa in chatDataId])+1)) for i in range(self.wordNum)], dtype='float32')

        print("Total sample num:",self.totalSampleNum)

        #划分训练和测试数据集
        self.trainIdList, self.testIdList = train_test_split([i for i in range(self.totalSampleNum)], test_size=testSize)
        self.trainSampleNum, self.testSampleNum = len(self.trainIdList), len(self.testIdList)
        print("train size: %d; test size: %d"%(self.trainSampleNum, self.testSampleNum))
        self.testSize = testSize
        print("Finished loading corpus!")


   #重置词的ID和映射关系
    def reset_word_id_map(self, id2word, word2id):
        self.id2word, self.word2id = id2word, word2id
        chatDataId = [[[self.word2id[w] for w in qa[0]],[self.word2id[w] for w in qa[1]]] for qa in self.chatDataWord]
        self.QChatDataId, self.AChatDataId = [qa[0] for qa in chatDataId], [qa[1] for qa in chatDataId]
    
    # 数据随机打乱，随机数据流
    def random_batch_data_stream(self, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, type='train'):
        #判断数据是不是训练数据，另外给定结束符
        idList = self.trainIdList if type=='train' else self.testIdList
        eosToken, unkToken = self.word2id['<EOS>'], self.word2id['<UNK>']
        while True:
            samples = random.sample(idList, min(batchSize, len(idList))) if batchSize>0 else random.sample(idList, len(idList))
            #数据增广
            if isDataEnhance:
                yield self._dataEnhance(samples, dataEnhanceRatio, eosToken, unkToken)
            else:
                QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
                QDataId = np.array([self.QChatDataId[i]+[eosToken for j in range(QMaxLen-self.QLens[i]+1)] for i in samples], dtype='int32')
                ADataId = np.array([self.AChatDataId[i]+[eosToken for j in range(AMaxLen-self.ALens[i]+1)] for i in samples], dtype='int32')
                yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]
    
    #确定一个epoch数据流 确认数据是训练数据
    def one_epoch_data_stream(self, batchSize=128, isDataEnhance=False, dataEnhanceRatio=0.2, type='train'):
        #判断数据是不是训练数据，另外给定结束符
        idList = self.trainIdList if type=='train' else self.testIdList
        eosToken = self.word2id['<EOS>']
        #做循环
        for i in range((len(idList)+batchSize-1)//batchSize):
            samples = idList[i*batchSize:(i+1)*batchSize]
            # 数据增广
            if isDataEnhance:
                yield self._dataEnhance(samples, dataEnhanceRatio, eosToken, unkToken)
            else:
                QMaxLen, AMaxLen = max(self.QLens[samples]), max(self.ALens[samples])
                QDataId = np.array([self.QChatDataId[i]+[eosToken for j in range(QMaxLen-self.QLens[i]+1)] for i in samples], dtype='int32')
                ADataId = np.array([self.AChatDataId[i]+[eosToken for j in range(AMaxLen-self.ALens[i]+1)] for i in samples], dtype='int32')
                yield QDataId, self.QLens[samples], ADataId, self.ALens[samples]
    #遍历
    def _purify(self, txt):
        return [filter_sent(qa) for qa in txt]
    #数据长度判断
    def _QALens(self, data):
        QLens, ALens = [len(qa[0])+1 for qa in data], [len(qa[1])+1 for qa in data]
        QMaxLen, AMaxLen = max(QLens), max(ALens)
        print('QMAXLEN:',QMaxLen,'  AMAXLEN:',AMaxLen)
        self.QLens, self.ALens = np.array(QLens, dtype='int32'), np.array(ALens, dtype='int32')
        self.QMaxLen, self.AMaxLen = QMaxLen, AMaxLen

    # word2id映射关系
    def _word_id_map(self, data):
        if self.id2word==None: 
            self.id2word = list(set([w for qa in data for sent in qa for w in sent]))
            self.id2word.sort()
            self.id2word = ['<EOS>','<SOS>'] + self.id2word + ['<UNK>']
        if self.word2id==None: self.word2id = {i[1]:i[0] for i in enumerate(self.id2word)}
        if self.wordNum==None: self.wordNum = len(self.id2word)
        print('Total words num:',len(self.id2word)-2)

    #数据增广的方式
    def _dataEnhance(self, samples, ratio, eosToken, unkToken):

       #计算tf_idf
        q_tf_idf = [[self.QIDF[w]*self.QChatDataId[i].count(w)/(self.QLens[i]-1) for w in self.QChatDataId[i]] for i in samples]


        q_tf_idf = [[j/sum(i) for j in i] for i in q_tf_idf]

        QDataId = [[w for cntw,w in enumerate(self.QChatDataId[i]) if random.random()>ratio or random.random()<q_tf_idf[cnti][cntw]] for cnti,i in enumerate(samples)]

       #随机一些ID
        QDataId = [[w if random.random()>ratio/5 else unkToken for w in self.QChatDataId[i]] for i in samples]

        ADataId = [self.AChatDataId[i] for i in samples]
       #然后遍历取其中的一些ID
        for i in range(len(samples)):
            if random.random()<ratio:random.shuffle(QDataId[i])

        QLens = np.array([len(q)+1 for q in QDataId], dtype='int32')
        ALens = np.array([len(a)+1 for a in ADataId], dtype='int32')
        QMaxLen, AMaxLen = max(QLens), max(ALens)

        QDataId = np.array([q+[eosToken for j in range(QMaxLen-len(q))] for q in QDataId], dtype='int32')
        ADataId = np.array([a+[eosToken for j in range(AMaxLen-len(a))] for a in ADataId], dtype='int32')

        validId = (QLens>1) & (ALens>1)
        return QDataId[validId], QLens[validId], ADataId[validId], ALens[validId]

#seq2id
def seq2id(word2id, seqData):
    seqId = [word2id[w] for w in seqData]
    return seqId

#id2seq
def id2seq(id2word, seqId):
    seqData = [id2word[i] for i in seqId]
    return seqData

#去掉一些停用词，替换中文的一些符号
def filter_sent(sent):
    return sent.replace('\n','').replace(' ','').replace('，',',').replace('。','.').replace('；',';').replace('：',':').replace('？','?').replace('！','!').replace('“','"').replace('”','"').replace("‘","'").replace("’","'").replace('（','(').replace('）',')')