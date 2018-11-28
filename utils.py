import numpy as np


# calculate Ecli distance
def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


# randomly choose k center
def randCenter(dataSetIn, kIn):
    while True:
        centroids = dataSetIn[np.random.choice(range(len(dataSetIn)), kIn, replace=False)]
        uniques = np.unique(centroids)
        if len(centroids) == len(uniques) / 2:
            break
    print("initialized centroids to ")
    print(centroids)
    return centroids


# do cluster
def kMeans(dataSetIn, kIn, syslog):
    # row of dataset,dataset is [x,y]
    # m = np.shape(dataSet)[0]
    m = len(syslog)
    # create matrix [cluster,distance, index of syslog] like [0,19, 2]
    clusterAssment = np.mat(np.zeros((m, 3)))
    centriods = randCenter(dataSetIn, kIn)
    clusterChange = True
    while clusterChange:
        clusterChange = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(kIn):
                distJI = distEclud(centriods[j, 0:1], dataSetIn[i, 0:1])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
            clusterAssment[i, :] = minIndex, minDist, i

        for cent in range(kIn):
            count = 0
            meanX = 0
            meanY = 0
            for i in range(m):
                if clusterAssment[i, 0] == cent:
                    meanX += dataSetIn[i, 0]
                    meanY += dataSetIn[i, 1]
                    count += 1
            centriods[cent] = np.array([meanX / count, meanY / count])
    return centriods, clusterAssment
