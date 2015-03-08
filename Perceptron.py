import pickle,random,copy
import numpy as np, matplotlib.pyplot as plt

class Perceptron:
    def __init__(self,Data,LearningRate):
        self.Vec = Data['vectors']
        self.Labels =Data['labels']
        self.Iteration = 0
        self.W = np.array([random.uniform(-1,1) for _ in range(2)])
        self.W1 = copy.copy(self.W)
        self.LearningRate = LearningRate

    def Response(self,Vec):
        '''Checks where the point lies in relation to the weight vector and classifies
        it as either 0 or 1'''
        return Vec[0]*self.W[0]+Vec[1]*self.W[1] < .5

    def UpdateWeight(self,Vec,d):
        '''Updates the weight vector given a misclassified point and it's correct classification (d).
        Intakes 'Vec' as a 2D list in the form of [X,Y] and d, it's correct response as an int, or float. '''
        #d = Desired response (0 or 1)
        #r = Perceptron response (0 or 1)
        r=self.Response(Vec)
        self.W[0] += self.LearningRate*Vec[0]*(d-r)
        self.W[1] += self.LearningRate*Vec[1]*(d-r)

    def Learn(self,N):
        '''Applies the Perceptron learning algorithm to 'learn' to classify points'''
        for i in range(N):
            pt = random.randint(0,len(self.Vec[0])-1) # Randomly select a point to attempt to classify
            self.UpdateWeight([self.Vec[0][pt],self.Vec[1][pt]],self.Labels[pt])
            self.Iteration+=1

    def plot(self,Save=False):
        '''Plots the data alongside the weight transpose'''
        self.Colours = ['r' if _ == 0 else 'b' for _ in self.Labels]
        plt.scatter(self.Vec[0],self.Vec[1],c=self.Colours,s=40,marker=ur'$\u2665$',label = 'Labeled Data')
        plt.plot([-self.W[1],self.W[1]],[self.W[0],-self.W[0]],'k',label =r'$w^\tau$')
        plt.plot([-self.W1[1],self.W1[1]],[self.W1[0],-self.W1[0]],'--k',label =r'$w_0^\tau$') #Plots original weight vector
        plt.legend(loc='upper left')
        plt.xlim((min(self.Vec[0])*1.2,max(self.Vec[0])*1.2))
        plt.ylim((min(self.Vec[1])*1.2,max(self.Vec[1])*1.2))
        plt.title('Perceptron after %s iterations ' % (self.Iteration))
        if Save == True:
            plt.savefig('PerceptronIterationz%s.png'%(self.Iteration))

