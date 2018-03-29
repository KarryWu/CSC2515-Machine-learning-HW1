from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X,y,features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        plt.subplot(3, 5, i + 1)
        plot(X[:,i],y,'o',markersize = 2)
        plt.xlabel(features[i],fontsize = 8)   ##add x,y label
        plt.ylabel('Median Value',fontsize = 8)    ##add x,y label
        plt.xticks(fontsize = 8)     ##adjust fontsize of x,y label
        plt.yticks(fontsize = 8)     ##adjust fontsize of x,y label
    plt.tight_layout()
    plt.show()


def fit_regression(X,Y):
    from numpy.linalg import solve
    bias_column = ones(X.shape[0])
    new_X = np.column_stack([bias_column,X])    ##add bias term
    mat_X = mat(new_X)
    mat_Y = mat(Y).T
    xTx = (mat_X.T)*mat_X
    xTy = (mat_X.T)*mat_Y
    w = numpy.linalg.solve(xTx,xTy)        ##make the derivative of the loss function equals to 0, we get xTx*w=xTy
    return w
    raise NotImplementedError()

def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)

    #TODO: Split data into train and test
    random.seed(0)
    indice_train = random.choice(X.shape[0],round(X.shape[0]*0.8),replace=False)
    train_x = np.array([X[i,:] for i in indice_train])
    train_y = np.array([y[i] for i in indice_train])
    indice_tot = array(range(X.shape[0]))
    indice_test = np.delete(indice_tot,indice_train)
    test_x = np.array([X[j,:] for j in indice_test])
    test_y = np.array([y[j] for j in indice_test])


    # Fit regression model
    w = fit_regression(train_x, train_y)
    bias_column1 = ones(test_x.shape[0])   ##add zero column to test_x
    new_test_x = np.column_stack([bias_column1,test_x])
    predict_y = array(new_test_x * w)
    
    #Tabulet each feature along with its associated weight
    new_features = features[:,np.newaxis]  ##add an axis to feature_name
    tab_prmeter = np.column_stack([new_features,w[1:]])
    tab_prmeter = mat(tab_prmeter).T
    
    
    #Plot the fitting figure
    f = plt.figure(2)
    for i in range(test_x.shape[1]):
        ax = plt.subplot(3,5,i+1)
        ax.plot(test_x[:,i],test_y,'o',markersize = 2, color = "blue", label="Test")
        ax.plot(test_x[:,i],predict_y,'o',markersize =2, color = "red", label="Predict")
        plt.xlabel(features[i],fontsize = 8)   ##add x,y label
        plt.ylabel('Median Value',fontsize = 8)    ##add x,y label
        plt.xticks(fontsize = 8)     ##adjust fontsize of x,y label
        plt.yticks(fontsize = 8)     ##adjust fontsize of x,y label

    plt.legend(loc='lower right',bbox_to_anchor=(4, 0.5))
    plt.tight_layout()     ##make the figure tight layout
    plt.show()
    # Compute fitted values, MSE, etc.
    RMS = sqrt(((predict_y - test_y[:,newaxis])**2).mean())   ##calculate RMS
    print("RMS=",RMS)


    MAE = abs((predict_y-test_y[:,newaxis])).mean()  ##calculate MAE
    print("MAE=",MAE)

    MSEM =((predict_y - test_y[:,newaxis])**2).mean()    ##calculate MSEM
    print("MSEM=",MSEM)


    tot = sum((test_y[:,newaxis] - predict_y)**2)
    res = sum((test_y[:,newaxis] - tile([test_y.mean()],[test_y.shape[0],1]))**2)
    R2 = 1 - tot/res   ##calculate R2
    print("R2 Score=", R2)

if __name__ == "__main__":
    main()

