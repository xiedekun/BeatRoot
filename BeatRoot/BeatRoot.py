########################
# Code Author: Dekun Xie
# BeatRoot Algorithm for Beat-Tracking from Simon Dixon (2001) 
# ' Dixon, S. (2001). 
# Automatic extraction of tempo and beat from expressive performances. 
# International Journal of Phytoremediation, 21(1), 39–58.'
########################

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import maximum_filter1d as maxfilt

# Tempo Induction Stage: Onset Detection

def onsetDetection(fileName, windowTime=0.04, hopRatio=0.25, wd=21, thr=0.01, salienceToOne=False):
    '''
        Using Spectral Flux to dectect onsets
        parameter:
        
        fileName: audio file path
        windowTime： the time of window
        hopRatio: how many percents overlap to move
        wd: width of filter for median and maximum filtering
        thr: the threshold to select peaks
    '''
    snd, sr = librosa.load(fileName, sr=None)
    print(f'the song is {int(len(snd) / sr)} seconds')
    hopTime = windowTime * hopRatio
    print('Tempo Induction: Onset Detection starts...')
    hop = int(hopTime * sr)
    wlen = int(2**np.ceil(np.log2(sr * windowTime)))
    snd = np.concatenate([np.zeros(wlen//2), snd, (np.zeros(wlen // 2))])
    frameCount = int((np.floor(len(snd) - wlen) / hop + 1))

    odf = np.zeros(frameCount)
    window = np.hamming(wlen)
    
    lastFrame = np.zeros(wlen)
    
    # Spectral Flux
    for i in range(frameCount):
        start = i * hop
        frame = np.fft.fft(snd[start: start+wlen] * window)
        odf[i] = np.sum(np.max(np.abs(frame) - np.abs(lastFrame),0))
        lastFrame = frame

    # normalise
    mx = np.max(odf)
    if mx > 0:
        odf = odf / mx
        
        
    # Get onsets by selecting peak part
    # wd must be odd
    if wd % 2 == 0:
        wd += 1 

    # getOnset
    medFiltODF = odf - medfilt(odf, wd)
    maxFiltODF = maxfilt(medFiltODF, wd, mode='nearest', axis=0)
    threshold = [max(i, thr) for i in maxFiltODF]

    # get peak index
    peakIndices = np.nonzero(medFiltODF >= threshold)
    peakTimes = peakIndices[0] * hopTime
    salience = odf[peakIndices]

    if salienceToOne:
        salience[:] = 1
    
    t = np.arange(len(odf)) * hopTime
    plt.figure(figsize=(14,3))
    plt.plot(t, odf)
    plt.plot(t, medFiltODF, 'c')
    plt.plot(t, threshold, 'y')
    for j in peakTimes:
        plt.axvline(j, ymax=1, color='k')
    print('Tempo Induction: get onsets successfully')
    plt.show()
    
    return peakTimes, salience

    # Tempo Induction Stage: Clustering

# Cluster
class Cluster:
    def __init__(self, IOI):
        self.IOIs = IOI
        self.size = 1
        self.interval = np.mean(self.IOIs)
        self.score = 0
        
    def addIOIs(self, IOI, size=1):
        self.IOIs += IOI
        self.size += size
        self.interval = np.mean(self.IOIs)
    
def relationshipFactor(d):
    if 1<=d and d<=4:
        return 6 - d
    elif 5<=d and d<=8:
        return 1
    else:
        return 0 

def bestClusters(clusters, number):
    print(f'get {number} best ranked clusters: ')
    cluster = clusters
    cluster.sort(key=lambda Cluster: Cluster.score,reverse=True)
    
    return cluster[0:number]

def clustering(peakTimes, number=10) -> Cluster:
    print('Tempo Induction: Clustering starts...')
    clusterWidth = 25/1000 # 25 ms
    clusters = []
    print('Tempo Induction: generate clusters...it may cost time, please wait...')
    for event1 in peakTimes:
        for event2 in peakTimes:
            if event1 == event2:
                break # remove 0 value; I think this better
            IOI = [np.abs(event1 - event2)]
            isAdded = False
            for cluster in clusters:
                if np.abs(cluster.interval - IOI) < clusterWidth:
                    cluster.addIOIs(IOI)
                    isAdded = True
            if not isAdded:
                clusters.append(Cluster(IOI))

    for cluster1 in clusters:
        for cluster2 in clusters:
            if cluster1 != cluster2 and np.abs(cluster1.interval - cluster2.interval) < clusterWidth:
                cluster1.addIOIs(cluster2.IOIs, cluster2.size)
                clusters.remove(cluster2)

    for cluster1 in clusters:
        for cluster2 in clusters:
            for n in range(1,9):
                if np.abs(cluster1.interval - n * cluster2.interval) < clusterWidth:
                    cluster1.score += relationshipFactor(n) * cluster2.size

    clusters = bestClusters(clusters, number)
    print('Interval, Size, Score')
    for i,cluster in enumerate(clusters):
        print(round(cluster.interval,4), cluster.size, cluster.score)
    print('')
    return clusters

    # Beat Tracking Stage: Agent

class Agent:
    def __init__(self, interval, onsetsTime, score):
        self.beatInterval = interval
        self.prediction = onsetsTime + interval
        self.history = [onsetsTime]
        self.score = score
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

                    
def removeDuplicateAgents(agents):
    print('remove duplicate agents...')
    agentsCopy = agents.copy()
    lastAgent = Agent(0,0,0)
    for agent in agents:
        #print(agent,agent == lastAgent)
        if agent == lastAgent:
            agentsCopy.remove(agent)
        lastAgent = agent
    return agentsCopy

def bestAgents(agents, number=5):
    print(f'get {number} best ranked agents')
    agents.sort(key=lambda Agent: Agent.score,reverse=True)            
    agents = removeDuplicateAgents(agents)
    return agents[0:number]

def agentsProcess(peakTimes, salience, clusters, number=5) -> Agent:
    # initialise
    print('Agent Initialisation...')
    startupTime = 5 # 5s
    agents = []
    limitedTimeIndice = np.where(peakTimes < startupTime)
    for cluster in clusters:
        for i, event in enumerate(peakTimes[limitedTimeIndice]):
            agents.append(Agent(cluster.interval, event, salience[i]))

    # mainloop
    print('Agent Mainloop...')
    timeOut = 1
    tolPostRatio = 0.8 # default 0.4
    tolPreRatio = 0.5 # default 0.2
    tolInner = 0.01
    correctionFactor = 4 # how many beats to change tempo

    for j, event in enumerate(peakTimes):
        for i, agent in enumerate(agents):
            if event - agent.history[len(agent.history) - 1] > timeOut:
                agents.remove(agent)
            else:
                while agents[i].prediction + tolPostRatio*agents[i].beatInterval < event:
                    agents[i].prediction += agents[i].beatInterval
                if agents[i].prediction - tolPreRatio*agents[i].beatInterval <= event \
                                and agents[i].prediction + tolPostRatio*agents[i].beatInterval >=event:
                    if np.abs(agents[i].prediction - event) > tolInner:
                        agents.append(agents[i])
                    error = event - agents[i].prediction
                    relativeError = np.abs(error/agents[i].beatInterval)
                    agents[i].beatInterval += error/correctionFactor
                    agents[i].prediction = event + agents[i].beatInterval
                    agents[i].history.append(event)
                    agents[i].score += (1-relativeError/2) * salience[j]
        
    agents = bestAgents(agents, number)
    print('Beat Interval, History Length, Score')
    for i, agent in enumerate(agents):
        if i== 10:
            break
        print(round(agent.beatInterval,4),len(agent.history),round(agent.score,4))
    return agents

def beatTracker(fileName):
    peakTimes, gSalience = onsetDetection(fileName, salienceToOne=False)
    clusters = clustering(peakTimes)
    agents = agentsProcess(peakTimes, gSalience, clusters)
    return agents[0].history

if __name__ == '__main__':
    fileName = 'data/Albums-Ballroom_Classics4-14.wav'
    beats = beatTracker(fileName)
    print('beats is: \n', beats)