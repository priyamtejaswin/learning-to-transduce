import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits 
import sklearn.preprocessing 
from sklearn.utils import shuffle
from ltt.models import Model 
from ltt.layers import Dense, Tanh, Sigmoid, ReLU, MSE
from ltt.optimizers import SGD

def main():
    
    data, target = load_digits(return_X_y=True)
    data, target = shuffle(data, target)
    target = target.reshape(len(target), 1)
    enc = sklearn.preprocessing.OneHotEncoder() 
    enc.fit(target)
    target = enc.transform(target).toarray()
    data = data / 16.0 # VERY IMPORTANT: ALWAYS SCALE DATA 


    # import ipdb; ipdb.set_trace()
    
    loss = MSE("mse_loss")
    sgd_optimizer = SGD()
    sgd_optimizer.alpha = 0.1
    model = Model(name="mnist_test", loss_layer=loss, optimizer=sgd_optimizer) 

    model.add(Dense(n_in=64, n_out=32, name="dense1"))
    model.add(Sigmoid(name="act1"))
    model.add(Dense(n_in=32, n_out=10, name="dense2"))
    model.add(Sigmoid(name="act2"))

    from grad_check import gradient_check 
    model.feature_size = 64
    # gradient_check(model)

    for epoch in range(500):
        print("Epoch: {}".format(epoch))
        epoch_loss = [] 
        for start_idx in range(0, len(data), 25):

            #batching
            end_idx = min(len(data), start_idx + 25)
            batch_x = data[start_idx:end_idx, :] 
            batch_y = target[start_idx:end_idx, :]

            # forward -> backward -> loss
            _ = model.do_forward(batch_x) 
            batch_loss = model.do_loss(batch_y)
            model.do_backward() 
            model.do_update() 

            epoch_loss.append(batch_loss)

        print("Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
        model.optimizer.alpha = model.optimizer.alpha 

    # Predict 
    data_test, target_test = data[:200], target[:200] 
    y_preds = model.do_forward(data_test) 
    target_test = np.argmax(target_test, axis=1)
    y_preds     = np.argmax(y_preds, axis=1) 
    print((y_preds==target_test).mean())

    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()

