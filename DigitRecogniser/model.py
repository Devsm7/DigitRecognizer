import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MNISTModel:
        def __init__(self, input_size=784, hidden_size=10, output_size=10, alpha=0.1):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.alpha = alpha
            self.W1, self.b1, self.W2, self.b2 = self.init_params()

    
        def init_params(self):
            W1 = np.random.randn(self.hidden_size , self.input_size) * 0.01
            b1 = np.zeros((self.hidden_size , 1)) 
            W2 = np.random.randn(self.output_size , self.hidden_size) *0.01
            b2 = np.zeros((self.output_size,1)) 

            return W1 , b1 , W2 , b2
        
        @staticmethod
        def ReLU(Z):
            return np.maximum(0 , Z)

        @staticmethod
        def deriv_ReLU(Z):
            return Z > 0
        
        @staticmethod
        def softmax(Z):
            expZ = np.exp(Z - np.max(Z))  # for stability
            return expZ / np.sum(expZ, axis=0, keepdims=True)


        def forward_prop(self ,X):
            Z1 = self.W1.dot(X) + self.b1
            A1 = self.ReLU(Z1)
            Z2 = self.W2.dot(A1) + self.b2
            A2 = self.softmax(Z2)
            return Z1 , A1 , Z2 , A2
        
        @staticmethod
        def one_hot(Y):
            one_hot_Y = np.zeros((Y.size , Y.max() + 1))
            one_hot_Y[np.arange(Y.size) , Y] =  1
            one_hot_Y = one_hot_Y.T
            return one_hot_Y


        def back_prop(self, Z1, A1, Z2, A2, X, Y):
            m = Y.size
            one_hot_Y = self.one_hot(Y)

            dZ2 = A2 - one_hot_Y
            dW2 = (1 / m) * dZ2.dot(A1.T)
            db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = self.W2.T.dot(dZ2) * self.deriv_ReLU(Z1)
            dW1 = (1 / m) * dZ1.dot(X.T)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

            return dW1, db1, dW2, db2
        
        def update_params(self, dW1, db1, dW2, db2):
            self.W1 -= self.alpha * dW1
            self.b1 -= self.alpha * db1
            self.W2 -= self.alpha * dW2
            self.b2 -= self.alpha * db2

        def train(self, X, Y, iterations=500):
            for i in range(iterations+1):
                Z1, A1, Z2, A2 = self.forward_prop(X)
                dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, X, Y)
                self.update_params(dW1, db1, dW2, db2)

                if i % 50 == 0:
                    preds = self.get_predictions(A2)
                    acc = self.get_accuracy(preds, Y)
                    print(f"Iteration {i} - Accuracy: {acc:.2f}%")
                    print(self.W1[0,0], self.W1[9,50], self.W1[5,690])

                        
        def predict(self, X):
                """
                Predict labels for given input X.
                
                X shape: (num_features, num_samples)
                Returns: predicted labels, shape (num_samples,)
                """
                Z1, A1, Z2, A2 = self.forward_prop(X)
                return self.get_predictions(A2)
        
        def test_prediction(self , index , X , Y):
            current_image = X[: , index , None]
            prediction = self.predict(X[: , index , None])
            label = Y[index]
            print('Prediction: ' , prediction)
            print('Label: ' , label)

            current_image = current_image.reshape((28,28)) * 255
            plt.gray()
            plt.imshow(current_image , interpolation='nearest')
            plt.show()
        
        @staticmethod
        def get_predictions(A2):
            return np.argmax(A2 , 0)
        
        @staticmethod
        def get_accuracy(predictions , Y):
            return (np.sum(predictions == Y) / Y.size )*100




