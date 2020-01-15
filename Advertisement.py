import random
import re
import numpy as np
import feedparser

def textParse(bigString):
    list0fTokens=re.split(r'\W',bigString)
    return [tok.lower() for tok in list0fTokens if len(tok)>2]

#遍历词汇表中的每个词并统计它在文本中出现的次数，后根据出现次数从高到低对词典进行排序，返回最高的30个单词
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    import feedparser
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)

    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen));testSet=[]

    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat=[];trainClassses=[]

    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClassses.append(classList[docIndex])
    p0V,p1V,pSpam=trainNB0(np.array(trainMat),np.array(trainClassses))
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam)!=classList[docIndex]:
            errorCount+=1
    print("the error rate is:",float(errorCount)/len(testSet))
    return vocabList,p0V,p1V


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


def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList) #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my vocabulary!"%word)
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

def createVocabList(dataSet):
    vocabSet=set([])  #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document)  #创建两个集合的并集
    return list(vocabSet)

#显示地域相关的用词
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[];topSF=[]
    for i in range(len(p0V)):
        if(p0V[i]>-6.0):topSF.append((vocabList[i],p0V[i]))
        if(p1V[i]>-6.0):topNY.append((vocabList[i],p1V[i]))
    sortedSF=sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY=sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

if __name__ == '__main__':
    #因书本给的不能更换
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    vocabList,pSF,pNY=localWords(ny,sf)
    getTopWords(ny,sf)