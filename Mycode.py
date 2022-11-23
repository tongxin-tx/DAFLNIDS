# coding=utf-8
import csv
import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from sklearn.metrics import confusion_matrix
import logging
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout, Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.set_printoptions(threshold=10000000000000)


def load_traindata(path):
    file_path = path
    feature = []
    label = []
    with (open(file_path, 'r')) as data_from:
        csv_reader = csv.reader(data_from)
        for i in csv_reader:
            t = i[:78]
            t.append(0)
            t.append(0)

            if int(i[78]) == 0:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)

            elif int(i[78]) == 1:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 2:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 3:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 4:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 5:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 6:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
    return feature, label

def load_testdata():
    file_path = 'CICIDS2018Test/mtest.csv'
    feature = []
    label = []
    with (open(file_path, 'r')) as data_from:
        csv_reader = csv.reader(data_from)
        for i in csv_reader:
            t = i[:78]
            t.append(0)
            t.append(0)

            if int(i[78]) == 0:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)

            elif int(i[78]) == 1:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 2:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 3:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 4:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 5:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
            elif int(i[78]) == 6:
                feature.append(t)
                label_list = [0] * 7
                label_list[int(i[78])] = 1
                label.append(label_list)
    return feature, label

def split_client_dataset(num_clients: int, len_dataset: int, fixed_size: int=None):
    """Generate a index list based on number of clients and length of dataset.
      Args:
          num_clients: number of clients
          len_dataset: Number of samples in the whole dataset
          fixed_size: If setted, each client will only take a fixed number of samples.
      Returns:
          A nested list with index list for each client.
    """
    ind_list = np.linspace(0, len_dataset - 1, len_dataset).astype(np.int32)
    client_data_list = []
    size = int(len_dataset / num_clients)
    for client in range(num_clients):
        if fixed_size is not None:
            size = fixed_size
        data_list = np.random.choice(ind_list, size, replace=False)

        ind_list = [i for i in ind_list if i not in data_list]
        client_data_list.append(data_list)
    return client_data_list


def get_model():
    """Get a CNN model in keras."""
    model = Sequential()
    # model.add(Reshape((-1,6, 6,1)))
    model.add(Conv2D(32, kernel_size=(4, 5), strides=(1, 1),
                     input_shape=(8, 10, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    return model


def init(model,x_test,y_test):
    """In keras if you don't run a funcition of the model, the model's wight would be empty [0].
       This is only for weight initilization.
    """
    model.evaluate(x_test[0:1,...], y_test[0:1,...],verbose=0)


if __name__ == '__main__':

    x_train, y_train = load_traindata('CICIDS2018Train/train.csv')
    x_test, y_test =load_testdata()


    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.fit_transform(x_test)


    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    t_round = 20
    numclients=2
    num_clients = 3
    len_dataset = x_train.shape[0]
    print(x_train.shape[0])
    print(x_train[0])
    #print(len(x_train[0]))

    x_train = x_train.reshape(x_train.shape[0], 8, 10, 1)
    x_test = x_test.reshape(x_test.shape[0], 8, 10, 1)

    global_model = get_model()
    print(global_model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.]))

    print(global_model.get_weights()[0].shape)
    print(global_model.get_weights()[1].shape)
    print(global_model.get_weights()[2].shape)
    print(global_model.get_weights()[3].shape)
    print(global_model.get_weights()[4].shape)
    print(global_model.get_weights()[5].shape)
    init(global_model, x_test, y_test)

    acc_list = []
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    acc5 = []
    acc6=[]
    acc7=[]

    #分割数据集
    client_data_list = split_client_dataset(2, len_dataset, fixed_size=None)

    x_train1, y_train1 = load_traindata('CICIDS2018Train/preDDos.csv')
    min_max_scaler1 = MinMaxScaler()
    x_train1 = min_max_scaler.fit_transform(x_train1)
    x_train1 = np.array(x_train1)
    y_train1 = np.array(y_train1)
    x_train1 = x_train1.reshape(x_train1.shape[0], 8, 10, 1)

    # x_train2, y_train2 = load_traindata('CICIDS2018Train/preInfilteration.csv')
    # x_train2 = min_max_scaler.fit_transform(x_train2)
    # x_train2 = np.array(x_train2, dtype=object)
    # y_train2 = np.array(y_train2, dtype=object)
    # x_train2 = x_train2.reshape(x_train2.shape[0], 8, 10, 1)
    #
    # x_train3, y_train3 = load_traindata('CICIDS2018Train/preBenign.csv')
    # x_train3 = min_max_scaler.fit_transform(x_train3)
    # x_train3 = np.array(x_train3, dtype=object)
    # y_train3 = np.array(y_train3, dtype=object)
    # x_train3 = x_train3.reshape(x_train3.shape[0], 8, 10, 1)



    for r in range(t_round):
        print("Round: "+str(r+1)+" started.")

        acc=[]
        # Size of weight based on the model
        weight_acc = np.asarray([np.zeros((4, 5, 1, 32)), np.zeros(
            (32,)), np.zeros((384, 1024)), np.zeros((1024,)), np.zeros((1024, 7))
                                    , np.zeros((7,))
                                 ])
        sumw=[]
        numw = []
        for c in range(num_clients):
            model = get_model()
            init(model,x_test,y_test)
            model.set_weights(global_model.get_weights())

            if c==2:

                c_feature=x_train1
                c_label=y_train1
            # elif c==5:
            #     c_feature = x_train2
            #     c_label = y_train2
            # elif c==6:
            #     c_feature = x_train3
            #     c_label = y_train3
            elif c<=1:
                ind = client_data_list[c]
                c_feature = np.take(x_train, ind, axis=0)
                c_label = np.take(y_train, ind, axis=0)

            numw.append(len(c_feature))
            # Train client
            model.fit(x=c_feature, y=c_label, epochs=1, validation_split=0.1, batch_size=512, verbose=1)  # cnn

            param_after = np.asarray(model.get_weights())

            sumw.append(param_after)

            score = model.evaluate(x_test, y_test, verbose=0)
            print('Client: '+str(c+1)+' with accuracy:', score[1])
            if c==0: acc1.append(score[1])
            elif c==1: acc2.append(score[1])
            elif c == 2:acc3.append(score[1])
            elif c == 3: acc4.append(score[1])
            elif c == 4: acc5.append(score[1])
            elif c == 5: acc6.append(score[1])
            elif c == 6: acc7.append(score[1])
            acc.append(score[1])

        nacc=[]
        nsumw=[]

        for c2 in range(num_clients):
            if acc[c2]>=0.75:
                nacc.append(np.exp(acc[c2]))
                nsumw.append(sumw[c2])


        weight = []
        accw=[]
        for c1 in range(len(nacc)):
            print(numw[c1]/sum(numw))
            print(nacc[c1]/sum(nacc))
            weight.append((nacc[c1]/sum(nacc))*(numw[c1]/sum(numw)))

        for c1 in range(len(nacc)):
            print(weight[c1] / sum(weight))
            weight_acc += nsumw[c1] *(weight[c1]/sum(weight))


        global_model.set_weights(weight_acc)
        score = global_model.evaluate(x_test, y_test, verbose=0)
        print('Global test loss:', score[0])
        print('Global accuracy:', score[1])
        acc_list.append(score[1])

    preds = global_model.predict(x_test)
    pred_lbls = np.argmax(preds,axis=1)
    true_lbls = np.argmax(y_test,axis=1)
    #print(pic_client)
    confusion_matrix(true_lbls, pred_lbls)

    print("confusion_matrix",confusion_matrix(true_lbls, pred_lbls))

    from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score
    sumf1=f1_score(true_lbls, pred_lbls, average='weighted')
    print("***********")
    print('f1',sumf1)
    print('recall',recall_score(true_lbls, pred_lbls, average='weighted'))
    print('accuracy',accuracy_score(true_lbls, pred_lbls))
    print('precision',precision_score(true_lbls, pred_lbls, average='weighted'))



    from sklearn.metrics import multilabel_confusion_matrix
    conf = multilabel_confusion_matrix(true_lbls, pred_lbls)

    print("***********")
    print("confusion_matrix",conf)
    class1=['Normal','Infilteration','Bot','DoS','SQL Injection','Brute Force','DDos']
    for i in range(conf.shape[0]):

        tn = conf[i][0][0]
        fp = conf[i][0][1]
        fn = conf[i][1][0]
        tp = conf[i][1][1]
        acc = (tp+tn)/(tp+tn+fp+fn)
        fpr = fp/(fp+tn)
        tpr = tp/(tp+fn)

        Precision=tp/(tp+fp)
        f1=(2*Precision*tpr)/(Precision+tpr)
        #print(conf[i])
        print(class1[i])
        print("Accuracy",float("{0:5f}".format(acc))*100)
        print("Precision",float("{0:5f}".format(Precision))*100)
        print("Recall", float("{0:5f}".format(tpr)) * 100)
        print("f1", float("{0:5f}".format(f1)) * 100)

    print(acc_list)
