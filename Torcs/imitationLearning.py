#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import random
import numpy as np
import cv2
import os
import pickle
from collections import deque


import json
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD , Adam, RMSprop
from gym_torcs import TorcsEnv


learning_rate = 1e-2
img_dim = [64, 64, 3]
n_action = 1        # steer only (float, left and right 1 ~ -1)
steps = 1000        # maximum step for a game
batch_size = 32     # for collecting imitation data
n_epoch = 100      # for training the model
n_episode = 5       # for retrain
memory = 10000
Imgpath = "./trainImg" 
parent_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(parent_path, 'trainedData')


class Agent():
    def __init__(self):
        print("Successfully get agent for learning")
        self.D = deque()

    def get_teacher_action(self, ob):
        """ Compute steer from image for getting data of demonstration """
        steer = ob.angle*10/np.pi
        steer -= ob.trackPos*0.10
        return np.array([steer])

    def img_reshape(self, input_img):
        """ (3, 64, 64) --> (1, 64, 64, 3) """
        _img = np.transpose(input_img, (1, 2, 0))  
        _img = np.flipud(_img)
        _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
        return _img

    def save_img(self, input_img, i):  # save the ith image collected for training
        if not os.path.exists(Imgpath):
            os.makedirs(Imgpath)
        
        img_ = np.transpose(input_img, (1, 2, 0))
        img = np.flipud(img_)
        cv2.imwrite(Imgpath + "/%d.jpg" % i, img)

    # build network
    def buildModel(self):
        print("Now we build the model")
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation = 'relu',input_shape=(64, 64, 3), kernel_initializer="glorot_normal")) 
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation = 'relu', kernel_initializer="glorot_normal"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation = 'relu',kernel_initializer="glorot_normal"))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation = 'relu',kernel_initializer="glorot_normal"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu' ,kernel_initializer="glorot_normal"))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_action, activation='tanh',kernel_initializer="glorot_normal"))
        self.model.add(BatchNormalization())

        adam = Adam(lr=learning_rate)
        self.model.compile(loss='mse',optimizer=adam)
        print("We finish building the model")


    def collectData(self):
    # collect state-action pairs of imitation learning
        # self.images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
        # self.actions_all = np.zeros((0, n_action))
        # self.rewards_all = np.zeros((0,))

        # img_list = []
        # action_list = []
        # reward_list = []
      
        self.env = TorcsEnv(vision=True, throttle=False)
        ob = self.env.reset(relaunch=True)

        print('Collecting data from expert ... ')
        for i in range(steps):
            if i == 0:
                act = np.array([0.0])
            else:
                act = self.get_teacher_action(ob)
                print("act %f" % act)
            if i % 100 == 0:
                print("step:", i)
            ob, reward, done, _ = self.env.step(act)
          #  img_list.append(ob.img/255)  # normanize RGB value to [0,1]
          #  action_list.append(act)
          #  reward_list.append(np.array([reward]))
            if i % 10 == 0:
                self.save_img(ob.img, i)
            self.D.append([self.img_reshape(ob.img/255), act, np.array([reward])])

            if len(self.D) > memory:
                self.D.popleft()

        print("step: %d" % steps)
        self.env.end()
        '''
        for img, act, rew in zip(img_list, action_list, reward_list):
            self.images_all = np.concatenate([self.images_all, self.img_reshape(img)], axis=0)
            self.actions_all = np.concatenate([self.actions_all, np.reshape(act, [1,n_action])], axis=0)
            self.rewards_all = np.concatenate([self.rewards_all, rew], axis=0)
        '''

    # pretrain
    def train(self):
        print("start train the model ...")
        for i in range(n_epoch):
            # lenth = len(self.images_all)
            # print("total image %d" % lenth)
            # trainIndex = random.sample(range(lenth), batch_size)
            # trainX = []
            # trainY = []
            # for j in trainIndex:
            #    trainX.append(self.images_all[j])
            #    trainY.append(self.actions_all[j])
            
            trainX_ = np.zeros((batch_size, img_dim[0], img_dim[1], img_dim[2]))
            trainY = np.zeros(batch_size)
            trainBatch = random.sample(list(self.D), batch_size)
            loss = 0
            for k in range(batch_size):               
                trainX_[k:k+1] = trainBatch[k][0]
                trainY[k:k+1] = trainBatch[k][1]
               
                # trainX_ = trainBatch[k][0]
                # trainY = trainBatch[k][1]
                # self.model.fit(trainX_, trainY, verbose=0)
                # self.model.fit(trainX_, trainY, verbose=0)
            #self.model.train_on_batch(trainX_, trainY)
            # self.model.train_on_batch(trainX_, trainY)
            loss += self.model.train_on_batch(trainX_, trainY)
            '''
            trainX_ = np.array(trainX_)
            print(trainX_.shape)
            trainX = []
            trainX.append(trainX_)
            trainX = np.array(trainX)
            '''
                
            print("Epoch: [%d/%d], loss=%f" % (i, n_epoch, loss))             

    def retrain(self):
       for episode in range(n_episode):
           self.env = TorcsEnv(vision=True, throttle=False)
           ob = self.env.reset(relaunch=True)
           reward_sum = 0
           i = 0
           print("# Episode: %d start" % episode)
           for i in range(steps):
              act = self.model.predict(self.img_reshape(ob.img/255))
              ob, reward, done, _ = self.env.step(act)
              if done is True:
                 break
              else:
                 self.D.append([self.img_reshape(ob.img/255), act, np.array([reward])])
              reward_sum += reward

           print("# step: %d reward: %f " % (i, reward_sum))
           self.env.end()
           if i == (steps-1):
              break
           self.train()
           self.save()     
       


    def save(self):
        print("Now we save model")
        self.model.save("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    def predict(self, image):
        a = self.model.predict(image)
        return a


if __name__ == '__main__':
    agent = Agent()
    agent.buildModel()
    if not os.path.exists(data_path):
       os.makedirs(data_path)
    data = os.path.join(data_path, 'data.pkl')
    # agent.collectData()
    # agent.model = load_model('model.h5')
    with open(data, 'rb') as f:
       agent.D = pickle.load(f)
    agent.train()
    '''
    with open(data,'ab+') as f:
       pickle.dump(agent.D, f, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    agent.save()
    learning_rate = 1e-4
    agent.retrain()

    # after train, just run 
 
    for i in range(1):
         env = TorcsEnv(vision=True, throttle=False)
         ob = env.reset(relaunch=True)
         # load the model
         model = load_model('model.h5')
         reward_sum = 0.0
         for i in range(steps):
             act = model.predict(agent.img_reshape(ob.img/255))
             ob, reward, done, _ = env.step(act)
             if done is True:
                 break
             reward_sum += reward
         print("PLAY WITH THE TRAINED MODEL")
         print(reward_sum)
         env.end()
    


