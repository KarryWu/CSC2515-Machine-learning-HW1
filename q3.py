import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)

    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))
    return dot / (sum1 * sum2)

#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    
    mat_X = np.mat(X)
    mat_y = np.mat(y).T
    mat_w = np.mat(w).T
    gd = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for i in range(X.shape[0]):
            ##print(mat_y[i]-(mat_w.T) * mat_X[i].reshape(13,1))
            gd[j] = gd[j] + (-2/X.shape[0])*float((mat_y[i] - (mat_w.T) * mat_X[i].reshape(13,1))*mat_X[i,j])
            
    return gd
    ##raise NotImplementedError()
    

##define a function to present the relationship between m and variance
def var_to_m(X,y,w):
    variance = np.zeros(400)
    for batch in range(1,401):
        batch_sampler3 = BatchSampler(X, y, batch)
        new_batch_grad = np.zeros(1)
        grad = np.zeros(500)
        for i in range(500):
            X_b3, y_b3 = batch_sampler3.get_batch()
            mat_X = np.mat(X_b3)
            mat_y = np.mat(y_b3).T
            mat_w = np.mat(w).T
            gd = 0
            for j in range(X_b3.shape[0]):
                gd = gd + 2/X_b3.shape[0]*float((mat_y[j] - (mat_w.T) * mat_X[j].reshape(13,1))*mat_X[j,0])
            grad[i] = gd
        variance[batch-1] = var(grad)
        print(batch)
    m1 = np.array(range(1,401))
    plt.plot(log(m1),log(variance))
    plt.xlabel("m")
    plt.ylabel("variance")
            
        
        

def main():
    # Load data and randomly initialise weights
    X, y, w = load_data_and_init_params()
    # Calculate true gradient
    batch_sampler1 = BatchSampler(X, y, X.shape[0])
    true_grad = np.zeros(X.shape[1])
    X_b1,y_b1 = batch_sampler1.get_batch()
    true_grad = lin_reg_gradient(X_b1, y_b1 ,w)
    print(true_grad)         ##check the value
    
    
    ##Calculate mini-batch gradient
    batch_sampler2 = BatchSampler(X, y, 50)
    new_batch_grad = np.zeros(X.shape[1])
    for i in range(500):
        X_b2, y_b2 = batch_sampler2.get_batch()
        batch_grad = lin_reg_gradient(X_b2, y_b2, w)
        new_batch_grad = new_batch_grad + batch_grad
    new_batch_grad = new_batch_grad/500
    print(new_batch_grad)    ##check the value
    
    square_dist =sum((true_grad - new_batch_grad)**2)
    print(square_dist)      ##get the value
    cosine_simi = cosine_similarity(true_grad, new_batch_grad)
    print(cosine_simi)      ##get the value
    
    var_to_m(X,y,w)    
    
    

if __name__ == '__main__':
    main()


