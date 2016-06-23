# -*- coding: utf-8 -*-
#样本是32*32的二值图片，将其处理成1*1024的特征行向量

import numpy 
import operator
from PIL import Image
from os import listdir

def img2vector(filename):
    #returnVect = zeros((1,400))   #产生一个数组将来用于存储图片数据

    img = Image.open(filename)
    img_ndarray=numpy.asarray(img,dtype='float64')
    returnVect=numpy.ndarray.flatten(img_ndarray)

#    for i in range(32):
#        lineStr = fr.readline()    #读取第一行的32个二值数据
#        for j in range(32):
#            returnVect[0,32*i+j] = int(lineStr[j])   #以32为一个循环 逐渐填满1024个
    return returnVect



def handwritingClassTest():
    #加载训练集到大矩阵trainingMat
    hwLabels = []
    trainingFileList = listdir('C:\\Anaconda\\trainingabc')           #os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表
    m = len(trainingFileList)                 #m表示总体训练样本个数
    trainingMat = numpy.zeros((m,400))             #用来存放m个样本
    for i in range(m):
        fileNameStr = trainingFileList[i]                  #训练样本的命名格式：1_120.txt 获取文件名
        fileStr = fileNameStr.split('.')[0]                #string.split('str')以字符str为分隔符切片，返回list，这里去list[0],得到类似1_120这样的
        classNumStr = str(fileStr.split('-')[0])           #以_切片，得到1，即数字类别
        hwLabels.append(classNumStr)                    #这样hwLabels列表中保存了m个样本点的所有类别
        trainingMat[i,:] = img2vector('C:\\Anaconda\\trainingabc\\%s' % fileNameStr)    #将每个样本存到m*1024矩阵
        
    #逐一读取测试图片，同时将其分类   
    testFileList = listdir('C:\\Anaconda\\testabc')     #测试文件名列表   
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]            #获取此刻要测试的文件名
        fileStr = fileNameStr.split('.')[0]     
        classNumStr = str(fileStr.split('-')[0])          #数字标签分类 重新生成的classNumStr 并没有使用上面已有的classNumStr
        vectorUnderTest = img2vector('C:\\Anaconda\\testabc\\%s' % fileNameStr)  #将要测试的数字转化为一行向量
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 5)   #传参
        print "the classifier came back with: %s, the real answer is: %s" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



#分类主程序，计算欧式距离，选择距离最小的前k个，返回k个中出现频次最高的类别作为分类别
#inX是所要测试的向量
#dataSet是训练样本集，一行对应一个样本，dataSet对应的标签向量为labels
#k是所选的最近邻数
def classify0(inX, dataSet, labels, k):      #参数和上面的一一对应
    dataSetSize = dataSet.shape[0]                       #shape[0]得出dataSet的行数，即训练样本个数
    diffMat = numpy.tile(inX, (dataSetSize,1)) - dataSet       #tile(A,(m,n))将数组A作为元素构造m行n列的数组 100*1024的数组
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)                  #array.sum(axis=1)按行累加，axis=0为按列累加
    distances = sqDistances**0.5                        #就是sqrt的结果
    sortedDistIndicies = distances.argsort()             #array.argsort()，得到每个元素按次排序后分别在原数组中的下标 从小到大排列
    classCount={}                                        #sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]   #最近的那个在原来数组中的下标位置
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #一个字典从无到有的生成过程  get(key,x)从字典中获取key对应的value，没有key的话返回0
    #classCount的形式:{5:3,0:6,1:7,2:1}
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) #sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
                                                         #sorted第一个参数表示要排序的对象 iteritems代表键值对 
    return sortedClassCount[0][0]                        #经过sorted后的字典变成了[(),()]形式的键值对列表




import numpy

import knnpabc

knnpabc.handwritingClassTest()
the classifier came back with: A, the real answer is: A
the classifier came back with: A, the real answer is: A
the classifier came back with: A, the real answer is: A
the classifier came back with: A, the real answer is: A
the classifier came back with: A, the real answer is: A
the classifier came back with: B, the real answer is: B
the classifier came back with: B, the real answer is: B
the classifier came back with: B, the real answer is: B
the classifier came back with: B, the real answer is: B
the classifier came back with: B, the real answer is: B
the classifier came back with: C, the real answer is: C
the classifier came back with: C, the real answer is: C
the classifier came back with: C, the real answer is: C
the classifier came back with: C, the real answer is: C
the classifier came back with: C, the real answer is: C
the classifier came back with: B, the real answer is: D
the classifier came back with: B, the real answer is: D
the classifier came back with: D, the real answer is: D
the classifier came back with: B, the real answer is: D
the classifier came back with: D, the real answer is: D
the classifier came back with: E, the real answer is: E
the classifier came back with: E, the real answer is: E
the classifier came back with: E, the real answer is: E
the classifier came back with: E, the real answer is: E
the classifier came back with: E, the real answer is: E
the classifier came back with: F, the real answer is: F
the classifier came back with: F, the real answer is: F
the classifier came back with: F, the real answer is: F
the classifier came back with: F, the real answer is: F
the classifier came back with: F, the real answer is: F
the classifier came back with: G, the real answer is: G
the classifier came back with: S, the real answer is: G
the classifier came back with: G, the real answer is: G
the classifier came back with: G, the real answer is: G
the classifier came back with: G, the real answer is: G
the classifier came back with: H, the real answer is: H
the classifier came back with: H, the real answer is: H
the classifier came back with: H, the real answer is: H
the classifier came back with: H, the real answer is: H
the classifier came back with: H, the real answer is: H
the classifier came back with: J, the real answer is: J
the classifier came back with: J, the real answer is: J
the classifier came back with: J, the real answer is: J
the classifier came back with: J, the real answer is: J
the classifier came back with: J, the real answer is: J
the classifier came back with: K, the real answer is: K
the classifier came back with: K, the real answer is: K
the classifier came back with: K, the real answer is: K
the classifier came back with: X, the real answer is: K
the classifier came back with: K, the real answer is: K
the classifier came back with: L, the real answer is: L
the classifier came back with: L, the real answer is: L
the classifier came back with: L, the real answer is: L
the classifier came back with: L, the real answer is: L
the classifier came back with: L, the real answer is: L
the classifier came back with: M, the real answer is: M
the classifier came back with: M, the real answer is: M
the classifier came back with: M, the real answer is: M
the classifier came back with: M, the real answer is: M
the classifier came back with: M, the real answer is: M
the classifier came back with: N, the real answer is: N
the classifier came back with: N, the real answer is: N
the classifier came back with: N, the real answer is: N
the classifier came back with: N, the real answer is: N
the classifier came back with: N, the real answer is: N
the classifier came back with: P, the real answer is: P
the classifier came back with: P, the real answer is: P
the classifier came back with: P, the real answer is: P
the classifier came back with: P, the real answer is: P
the classifier came back with: P, the real answer is: P
the classifier came back with: Q, the real answer is: Q
the classifier came back with: Q, the real answer is: Q
the classifier came back with: Q, the real answer is: Q
the classifier came back with: Q, the real answer is: Q
the classifier came back with: Q, the real answer is: Q
the classifier came back with: R, the real answer is: R
the classifier came back with: R, the real answer is: R
the classifier came back with: R, the real answer is: R
the classifier came back with: R, the real answer is: R
the classifier came back with: R, the real answer is: R
the classifier came back with: S, the real answer is: S
the classifier came back with: S, the real answer is: S
the classifier came back with: S, the real answer is: S
the classifier came back with: S, the real answer is: S
the classifier came back with: S, the real answer is: S
the classifier came back with: T, the real answer is: T
the classifier came back with: T, the real answer is: T
the classifier came back with: T, the real answer is: T
the classifier came back with: T, the real answer is: T
the classifier came back with: T, the real answer is: T
the classifier came back with: U, the real answer is: U
the classifier came back with: U, the real answer is: U
the classifier came back with: U, the real answer is: U
the classifier came back with: U, the real answer is: U
the classifier came back with: U, the real answer is: U
the classifier came back with: V, the real answer is: V
the classifier came back with: V, the real answer is: V
the classifier came back with: V, the real answer is: V
the classifier came back with: V, the real answer is: V
the classifier came back with: V, the real answer is: V
the classifier came back with: W, the real answer is: W
the classifier came back with: W, the real answer is: W
the classifier came back with: W, the real answer is: W
the classifier came back with: W, the real answer is: W
the classifier came back with: W, the real answer is: W
the classifier came back with: X, the real answer is: X
the classifier came back with: X, the real answer is: X
the classifier came back with: X, the real answer is: X
the classifier came back with: X, the real answer is: X
the classifier came back with: X, the real answer is: X
the classifier came back with: Y, the real answer is: Y
the classifier came back with: Y, the real answer is: Y
the classifier came back with: Y, the real answer is: Y
the classifier came back with: Y, the real answer is: Y
the classifier came back with: Y, the real answer is: Y
the classifier came back with: Z, the real answer is: Z
the classifier came back with: Z, the real answer is: Z
the classifier came back with: Z, the real answer is: Z
the classifier came back with: Z, the real answer is: Z
the classifier came back with: Z, the real answer is: Z

the total number of errors is: 5

the total error rate is: 0.041667

