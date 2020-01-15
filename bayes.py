#4.1
import numpy as np
import re
import random
###################################词表到向量的转换函数
def loadDataSet():
    postingList=[['my','dog','has','flea','problems','help','please'],
                 ['maybe','not','take','him','to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute','I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['mr','licks','ate','my','steak','how','to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]#1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])  #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document)  #创建两个集合的并集
    return list(vocabSet)

#该函数的输入参数为词汇表及某个文档，输出的是文档向量，向量的每一元素为0或1，分别表示词汇表中的单词在输入文档中是否出现
#函数首先创建一个和词汇表等长的向量，并将其元素设置为0，接着遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档像两种的对应值设为1
#如果一切都顺利的话，就不需要检查某个词时候还在vocabList中
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my vocabulary!"%word)
    return returnVec

##########################################朴素贝叶斯分类器训练函数
def  trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)#p0Num=np.zeros(numWords)
    p1Num=np.ones(numWords)#p1Num=np.zeros(numWords)
    p0Demo=2.0;p1Demo=2.0
    #p0Demo=0.0;p1Demo=0.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Demo+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Demo+=sum(trainMatrix[i])
    p1Vect=p1Num/p1Demo
    p0Vect=p0Num/p0Demo
    return p0Vect,p1Vect,pAbusive

#######################################朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    #listOPosts为文本内容，listClasses侮辱性文字向量
    listOPosts,listClasses=loadDataSet()
    #myVocabList为不重复词列表
    myVocabList=createVocabList(listOPosts)
    trainMat=[]

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print (testEntry , 'classified as: ' , classifyNB(thisDoc, p0V, p1V, pAb))

#####################################朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

#######################################文件解析及完整的垃圾邮件测试函数
def textParse(bigString):
    list0fTokens=re.split(r'\W',bigString)
    return [tok.lower() for tok in list0fTokens if len(tok)>2]

def spamTest():
    docList=[];classList=[];fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt'%i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]

    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]

    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("the error rate is :",float(errorCount)/len(testSet))

if __name__ == '__main__':
    spamTest()