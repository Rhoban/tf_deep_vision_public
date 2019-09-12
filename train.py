#!/usr/bin/env python

import tensorflow as tf
from tensorflow.python.framework import graph_util
import numpy as np
import os
from dataset import *
from model import *
import cv2
import utils
import datetime
import argparse

now = datetime.datetime.now()

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str

argParser = argparse.ArgumentParser(description='Training')
argParser.add_argument('-g', '--gpuNb', type=int, required=True, help="gpu id for training")
argParser.add_argument('-d', '--saveDirectory', type=typeDir, required=False, default="./", help="where to save the results")
argParser.add_argument('-ns', '--noSave', type=bool, required=False, default=False, help="do not save results")
argParser.add_argument('-ds', '--dataSet', type=str, required=True, help="dataset json file")
args = argParser.parse_args()

if (args.noSave == True):
    if (not utils.query_yes_no("/!\ WARNING, not saving, continue ?", default="no")):
        print("Exiting...")
        exit()


os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuNb)

dataset = Dataset()
dataset.fillDataset(args.dataSet)

batch_size = 64
nb_epochs = 100
learning_rate = 0.001

model = generic_cnn
model_params = {}
model_params["nb_conv_layers"] = 2 # number of convolution layers
model_params["nb_fc_layers"] = 1 # number of fully connected layers
model_params["nb_outputs"] = len(dataset.features_used)
model_params["conv_f"] = [8, 16] # number of features for each convolution layer
model_params["conv_k"] = [[5, 5], [3, 3]] # kernels for each convolution layer
model_params["conv_p"] = [[4, 4], [2, 2]] # poolings after each convolution level
model_params["conv_a"] = "identity" # activation
model_params["fc_l"] = [32] # number of neurons for each fully connected layer
model_params["fc_a"] = "identity" # activation

if(args.noSave == False):
        
        print("Name of the experiment : ")
        expName = input()
        if(expName==""):
                expName="model"
                
        print("Write a comment on the current experiment : ")
        comment = input()
        
        model_folder = args.saveDirectory+"/"+str(expName)+"_"+now.strftime("%Y-%m-%d_%H:%M")+"/"
        os.system("mkdir "+str(model_folder))
        os.system("unlink lastModel")
        os.system("ln -sf "+str(model_folder)+" lastModel")
        logs_dir = str(model_folder)+"/logs/"

        with open(str(model_folder)+"/model_params.json", 'w') as f:
                json.dump(model_params, f)

if __name__ == "__main__":

        nbBatchsInEpoch = dataset.buildBatches(batch_size)
        training_set, validation_set = dataset.getFilesSets()

        if(args.noSave == False):
                sets = {"training_set":training_set, "validation_set":validation_set}
        
                with open(str(model_folder)+"/sets.json", 'w') as f:
                        json.dump(sets, f)
                        
                with open(str(model_folder)+"/labels_map.json", 'w') as f:
                        json.dump(dataset.labels_map, f)
                        
                with open(str(model_folder)+"/inverted_labels_map.json", 'w') as f:
                        json.dump(dataset.inverted_labels_map, f)

                modelInfos = open(model_folder+"/infos", "a")
                
                modelInfos.write("Date : "+str(now.strftime("%Y-%m-%d_%H:%M"))+"\n\n")
                
                modelInfos.write("Comment : "+str(comment)+"\n\n")
        
                modelInfos.write("Model name : "+str(model.__name__)+"\n")
                modelInfos.write("Batch size : "+str(batch_size)+"\n")
                modelInfos.write("Image size : "+str(dataset.output_image_size)+"\n")
                modelInfos.write("Learning rate : "+str(learning_rate)+"\n\n")
                modelInfos.write("Train proportion : "+str(dataset.train_proportion)+"\n\n")
                modelInfos.write("Number of epochs : "+str(nb_epochs)+"\n\n")
                modelInfos.write("Model parameters : \n")
                modelInfos.write(str(model_params)+"\n\n")
                
                modelInfos.close()

        placeholder_shape = [None] + [dataset.output_image_size, dataset.output_image_size, 3]
            
        # placeholder_shape = [None] + [dataset.imShape]
        print("placeholder_shape", placeholder_shape)

        y_true = tf.to_float(tf.placeholder(tf.int32, shape=[None, len(dataset.labels_map)], name="y_true"))
        img_placeholder = tf.placeholder(tf.float32, placeholder_shape, name="img_placeholder")
        y_pred = model(img_placeholder, model_params)

        # Loss functions
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred))
        # cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        # cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=tf.concat([y_pred, tf.constant(len(dataset.labels_map))], axis=1)))
        loss = cross_entropy
        loss_summary = tf.summary.scalar("loss", loss) # tensorboard stuff

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.summary.scalar("accuracies", accuracy)
        
        # Optimizers
        # train_step = tf.train.MomentumOptimizer(learning_rate, 0.99, use_nesterov=False).minimize(loss)
        # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        currentEpoch = 0
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                # tensorboard stuff
                writer = tf.summary.FileWriter(logs_dir+"loss", sess.graph)
                writer1 = tf.summary.FileWriter(logs_dir+"validation_accuracy")
                writer2 = tf.summary.FileWriter(logs_dir+"train_accuracy")
                
                validationAcc = []
                trainAcc = []
                losses = []
                
                batch_validation_images, batch_validation_labels, batch_validation_paths = dataset.getValidation()
                
                i = 0
                while(currentEpoch < nb_epochs):
                        currentBatch, batch_train_images, batch_train_labels = dataset.getBatch()

                        _, l, summary = sess.run([train_step, loss, loss_summary], feed_dict={img_placeholder:batch_train_images, y_true: batch_train_labels})
                        writer.add_summary(summary, i)
                        
                        losses.append(l)

                        print("Epoch : "+str(currentEpoch)+", i : "+str(i)+", training loss   : "+str(round(l, 4))+", mega smoothed loss : "+str(round(np.mean(losses[-1000:]), 4)))
                        
                        if(currentBatch >= nbBatchsInEpoch-1):
                                nbBatchsInEpoch = dataset.buildBatches(batch_size)
                                currentEpoch+=1

                                validation_acc, summary = sess.run([accuracy, accuracy_summary], feed_dict={img_placeholder: batch_validation_images, y_true: batch_validation_labels})
                                writer1.add_summary(summary, currentEpoch)
                                validationAcc.append(validation_acc)
                                print("Validation accuracy : "+str(round(validation_acc, 4))+" smoothed Validation accuracy : "+str(round(np.mean(validationAcc[-10:]), 4)))
                                
                                train_acc, summary = sess.run([accuracy, accuracy_summary], feed_dict={img_placeholder: batch_train_images, y_true: batch_train_labels})
                                writer2.add_summary(summary, currentEpoch)
                                trainAcc.append(train_acc)
                                print("train accuracy : "+str(round(train_acc, 4))+" smoothed train accuracy : "+str(round(np.mean(trainAcc[-10:]), 4)))

                                sample = sess.run(y_pred, feed_dict={img_placeholder: batch_validation_images})
                                
                                print("=============================================")
                                print("sample :")

                                print(sample[0])
                                print(utils.softmax(sample[0]))
                                print(batch_validation_labels[0])
                                print("=============================================")
                                
                                print("")
                                print("Confusion Matrix (cols : prediction, rows : label)")
                                con_mat = tf.confusion_matrix(tf.argmax(batch_validation_labels, 1), tf.argmax(sample, 1))
                                confusion_matrix = tf.Tensor.eval(con_mat,feed_dict=None, session=None).astype(np.float64)
                                
                                print("")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        print(str(dataset.inverted_labels_map[str(j)])+" ", end=' ')
                                
                                print("")
                                print(confusion_matrix.astype(np.int32))
                                normal_confusion_matrix = confusion_matrix.astype(np.int32)
                                confusion_sum = np.sum(confusion_matrix)
                                for a in range(0, len(confusion_matrix[0])):
                                        for b in range(0, len(confusion_matrix[0])):
                                                confusion_matrix[a][b] = round(confusion_matrix[a][b]/confusion_sum, 3)

                                tmpLabels = []
                                print("")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        print(str(dataset.inverted_labels_map[str(j)])+" ", end=' ')
                                        tmpLabels.append(str(dataset.inverted_labels_map[str(j)]))
                                        
                                print("")
                                print(confusion_matrix)

                                colorMap = cv2.COLORMAP_JET
                                confusion_matrix *= 255
                                confusion_matrix = cv2.applyColorMap(confusion_matrix.astype(np.uint8), colorMap)
                                confusion_matrix = cv2.resize(confusion_matrix, (0, 0), fx=25, fy=25, interpolation=cv2.INTER_NEAREST)
                                
                                top_margin= 0
                                left_margin= 150
                                
                                blank_image = np.zeros((confusion_matrix.shape[0]+top_margin, confusion_matrix.shape[1]+left_margin, 3)).astype(np.uint8)
                                blank_image[:, :, :] = 255
                                blank_image[top_margin:top_margin+confusion_matrix.shape[0], left_margin:left_margin+confusion_matrix.shape[1], :] = confusion_matrix
                                
                                text_image = np.zeros((confusion_matrix.shape[0]+top_margin, confusion_matrix.shape[1]+left_margin, 3)).astype(np.uint8)
                                text_image[:, :, :] = 255
                                iii = 0
                                for l in tmpLabels:
                                    cv2.putText(text_image, l, (5, top_margin+iii*25+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                                    iii+=1


                                blank_image[0:blank_image.shape[0], 0:left_margin, :] = text_image[0:blank_image.shape[0], 0:left_margin, :]
                                epoch_folder = str(model_folder)+"/epoch_"+str(currentEpoch)
                                cv2.imwrite(str(epoch_folder)+"/confusion_matrix.png", blank_image)
                                print("WRITTEN")
                                
                                print("=============================================")
                                
                                print("========= Saving...=========")
                                saver.save(sess, epoch_folder+"/model.ckpt")
                                os.system("touch "+epoch_folder+"/infos")
                                os.system("unlink "+str(model_folder)+"/lastEpoch")
                                os.system("ln -sf epoch_"+str(currentEpoch)+" "+str(model_folder)+"/lastEpoch")

                                utils.visualize_false_predictions(epoch_folder,
                                                                  batch_validation_images,
                                                                  batch_validation_labels,
                                                                  batch_validation_paths,
                                                                  sample,
                                                                  dataset.inverted_labels_map)
                                
                                epochInfos = open(epoch_folder+"/infos", "a")
                                epochInfos.write("Epoch : "+str(currentEpoch)+"/"+str(nb_epochs)+"\n")


                                epochInfos.write("training Loss : "+str(round(np.mean(losses[-1000:]), 4))+"\n")
                                epochInfos.write("train accuracy : "+str(round(trainAcc[len(trainAcc)-1], 4))+"\n")
                                epochInfos.write("validation accuracy : "+str(round(validationAcc[len(validationAcc)-1], 4))+"\n\n")
                                
                                epochInfos.write("Confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        epochInfos.write(str(dataset.inverted_labels_map[str(j)])+" ")
                                epochInfos.write("\n")
                                epochInfos.write(str(normal_confusion_matrix)+"\n\n")
                                
                                epochInfos.write("Normalized confusion matrix : \n")
                                for j in range(0, len(dataset.inverted_labels_map)):
                                        epochInfos.write(str(dataset.inverted_labels_map[str(j)])+" ")
                                epochInfos.write("\n")
                                epochInfos.write(str(confusion_matrix)+"\n\n")
                                                                
                                epochInfos.close()

                                # exporting .pb file
                                print("========= Exporting .pb and .pbtxt...=========")
                                graph = tf.get_default_graph()
                                input_graph_def = graph.as_graph_def()

                                output_graph_def = graph_util.convert_variables_to_constants(
                                    sess,
                                    input_graph_def,
                                    ["generic_cnn/fully_connected/inference_output/Softmax"])
                                # [y_pred.name.split(":")[0]])
                                # ["tiny_model/output/output/Softmax"])  # WARNING change this if you change the model
                                
                                with tf.gfile.GFile(str(epoch_folder)+"/model.pb", "wb") as f:
                                        f.write(output_graph_def.SerializeToString())
                                        tf.train.write_graph(sess.graph_def, '.', str(epoch_folder)+'/model.pbtxt')
                                
                                        
                                
                        i+=1                                





