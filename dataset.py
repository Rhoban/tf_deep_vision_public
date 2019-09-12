#!/usr/bin/env python

import numpy as np
import cv2
import os
import random
import json
import math
import collections
import utils

class Dataset():

        def readImage(self, path):
                im = cv2.imread(path)
                if im is None:
                        return None
                im = cv2.resize(im, (self.output_image_size, self.output_image_size))
                # WARNING between -1 and 1, before it was between 0 and 1
                im = cv2.normalize(im, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                self.imShape = im.shape
        
                return im
        
        def getFilesSets(self):
                return self.training_set_paths, self.validation_set_paths
        
        def buildLabelsMapAndInvertedLabelsMap(self, features_used):
                print("== Building labels_map and inverted_labels_map ==")
                
                if "Empty" not in features_used:
                        features_used.append("Empty")
                        
                features_used = sorted(features_used) # sort alphabetically
                self.labels_map = {}
                self.inverted_labels_map = {}
                for i in range(0, len(features_used)):
                        self.labels_map[features_used[i]] = i
                        self.inverted_labels_map[i] = features_used[i]

                print(self.labels_map)
                print(self.inverted_labels_map)

        # centerDistThresholdMax       : Max allowed distance between center of patch and position of the feature in the patch
        #                                Warning : depends on the size of the patch
        # planarObjectDistThresholdMax : Max allowed distance to the camera for planar features (i.e everything but balls, posts and robots)
        def get_features(self, json_path, center_dist_threshold_max, planar_object_dist_threshold_max):
                json_file = open(json_path)
                data = json.load(json_file)


                features = np.zeros(len(self.labels_map))
                
                if str(data) == "null" or str(data) == "None":
                        features[self.labels_map["Empty"]] = 1
                        return features
                
                for key, value in data.items():
                        
                        i = 0
                        
                        if key not in self.labels_map:
                                continue

                        # Skipping null elements
                        while(str(value[i]) == "None"):
                                print("DEBUG None in Json : " + str(json_path))
                                i+=1

                        center = value[i]['center']

                        resize_factor = self.output_image_size/self.input_image_size
                        
                        center = np.array([float(center[0])*resize_factor, float(center[1])*resize_factor])
                        
                        img_center = np.array([self.output_image_size/2, self.output_image_size/2])
                        dist_to_center = np.linalg.norm(center-img_center)
                        
                        if dist_to_center > center_dist_threshold_max:
                                self.discarded_features[str(json_path)] = "Feature : "+str(key)+", distToCenter : "+str(dist_to_center)
                                continue

                        # if one of the features is too far, ignoring patch
                        if value[i]['distance'] > planar_object_dist_threshold_max and not key in ["Ball", "PostBase", "Robot"]:
                                self.discarded_patchs[str(json_path)] = "Feature : "+str(key)+", distToObj : "+str(value[i]['distance'])
                                return None
                        else:
                                features[self.labels_map[key]] = 1
                
                if 1 not in features:
                        self.discarded_patchs[str(json_path)] = "No features nor Empty"
                        return None                
                
                return features / sum(features)
                        
        def buildTrainingAndValidationSets(self):

                self.infos_by_path = {}
                
                json_paths = []
                for data_path in self.data_paths:
                        for file in os.listdir(data_path):
                                if "patch" in file:
                                        if file.endswith(".json"):
                                                json_paths.append(data_path+"/"+file)

                print("== Building infosByPath (this operation can take some time) == ")
                i = 0
                for json_path in json_paths:
                        percentage = int(i/len(json_paths)*100)
                        if i < len(json_paths)-1:
                                print("Treated "+str(percentage)+"% of the images", end="\r")
                        else:
                                print("Treated 100% of the images")
                                print("== DONE ==")
                                print("")

                        image_path = utils.getImg(json_path)

                        features = self.get_features(json_path, 8, 3)

                        if features is None:
                                continue

                        self.infos_by_path[image_path] = features
                        
                        i+=1

                print("======= DISCARDED PATCHS ===========")
                for k, v in self.discarded_patchs.items():
                        print(k, v)                        
                print(str(len(self.discarded_patchs)) + "/" + str(i))
                
                print("======= DISCARDED FEATURES ===========")
                for k, v in self.discarded_features.items():
                        print(k, v)
                print(str(len(self.discarded_features)) + "/" + str(i))

                total_features = np.zeros(len(self.labels_map))
                empty_patches = []
                for k, v in self.infos_by_path.items():
                        total_features += v
        
                        if v[self.labels_map["Empty"]] > 0:
                                empty_patches.append(k)
                                
                random.shuffle(empty_patches)

                # except empty
                max_occ_feature = 0
                for k, v in self.labels_map.items():
                        if k != "Empty" and total_features[v] > max_occ_feature:
                                max_occ_feature = int(total_features[v])
                      
                for e in empty_patches[max_occ_feature:]:
                        del self.infos_by_path[e]
                        
                self.infos_by_path =  utils.shuffleDict(self.infos_by_path)
                
                self.training_set = {}
                self.validation_set = {}
                
                i = 0
                for key, value in self.infos_by_path.items():
                        if i < self.train_proportion * len(self.infos_by_path): # TRAINING
                                self.training_set[key] = value
                                
                        else: # VALIDATION
                                self.validation_set[key] = value
                        i+=1

                if (os.path.exists("VISU") and not utils.query_yes_no("/!\ WARNING, VISU directory already exists. Overwrite ?", default="no")):
                        print("Skipping classes visualization")
                else:
                        os.system("rm -rf VISU")
                        os.system("mkdir VISU")
                        i = 0
                        for k, v in self.infos_by_path.items():
                                percentage = int(i/len(self.infos_by_path)*100)
                                if i < len(self.infos_by_path)-1:
                                        print("Treated "+str(percentage)+"% of the paths", end="\r")
                                else:
                                        print("Treated 100% of the paths")
                                        print("== DONE ==")
                                        print("")
                                
                                class_name = ""
                                iter = 0
                                for vv in v:
                                        if vv != 0:
                                                class_name += self.inverted_labels_map[iter]+'_'
                                        iter += 1
                                        
                                if class_name[len(class_name)-1] == '_':
                                        class_name = class_name[:-1]

                                os.system("mkdir -p VISU/"+str(class_name))

                                os.system("ln -sf ../../"+str(k)+" VISU/"+str(class_name)+"/"+str(i)+".png")

                                i+=1


                for k, v in self.training_set.items():                        
                        self.training_set_paths.append(k)
                
                for k, v in self.validation_set.items():
                        im = self.readImage(k)
                        if im is None:
                                continue
                                
                        
                        self.validation_set_paths.append(k)
                        self.validation_set_images.append(im)
                        self.validation_set_labels.append(v)
                                

        def buildBatches(self, batch_size):
                self.current_batch = 0
                all_images = []
                nb_batches = len(self.training_set)//batch_size

                for i in range(0, nb_batches):
                        self.batches.append([])

                current_batch_idx = 0
                i = 0
                for key, value in self.training_set.items():
                        if(current_batch_idx >= len(self.batches)):
                                break
                        self.batches[current_batch_idx].append([value, key])
                        if i!=0 and i%batch_size == batch_size-1:
                                current_batch_idx += 1
                        i+=1

                return nb_batches
                

        def getBatch(self):
                
                batch = self.batches[self.current_batch]
                labels = []
                images = []
                for b in batch:
                        labels.append(b[0])
                        im = self.readImage(b[1])
                        if im is None:
                                continue
                                
                        
                        images.append(im)

                tmp_images = np.ndarray(shape=(len(images), self.output_image_size, self.output_image_size, 3))
                for i in range(0, len(images)):
                        tmp_images[i] = images[i]
                        
                self.current_batch += 1
                
                return self.current_batch-1, tmp_images, labels

        def getValidation(self):
                return self.validation_set_images, self.validation_set_labels, self.validation_set_paths

        def computeStatsOnSet(self, merge=False):
                stats = {}

                stats["training_set"] = {}
                stats["validation_set"] = {}


                
                for k, v in self.training_set.items():
                        class_name = ""
                        iter = 0
                        for vv in v:
                                if vv != 0:
                                        class_name += self.inverted_labels_map[iter]+"_"
                                iter += 1
                                
                        if class_name[len(class_name)-1] == '_':
                                class_name = class_name[:-1]


                        if merge and '_' in class_name:
                                splitted = class_name.split('_')
                                if "mixed" not in stats["training_set"]:
                                        stats["training_set"]["mixed"] = 0
                                stats["training_set"]["mixed"] += len(splitted)

                                for s in splitted :
                                        if s not in stats["training_set"]:
                                                stats["training_set"][s] = 0
                                        stats["training_set"][s] += 1
                                        
                                        if s+"_in_mixed" not in stats["training_set"]:
                                                stats["training_set"][s+"_in_mixed"] = 0
                                        stats["training_set"][s+"_in_mixed"] += 1
                        else:
                                if class_name not in stats["training_set"]:
                                        stats["training_set"][class_name] = 0
                                stats["training_set"][class_name] += 1

                for k, v in self.validation_set.items():
                        class_name = ""
                        iter = 0
                        for vv in v:
                                if vv != 0:
                                        class_name += self.inverted_labels_map[iter]+"_"
                                        
                                iter += 1
                                
                        if class_name[len(class_name)-1] == '_':
                                class_name = class_name[:-1]


                        if merge and '_' in class_name:
                                splitted = class_name.split('_')
                                if "mixed" not in stats["validation_set"]:
                                        stats["validation_set"]["mixed"] = 0
                                stats["validation_set"]["mixed"] += len(splitted)

                                for s in splitted :
                                        if s not in stats["validation_set"]:
                                                stats["validation_set"][s] = 0
                                        stats["validation_set"][s] += 1
                                        
                                        if s+"_in_mixed" not in stats["validation_set"]:
                                                stats["validation_set"][s+"_in_mixed"] = 0
                                        stats["validation_set"][s+"_in_mixed"] += 1
                        else:
                                if class_name not in stats["validation_set"]:
                                        stats["validation_set"][class_name] = 0
                                stats["validation_set"][class_name] += 1

                return stats
                        
                
        
        
        def writeDataset(self):
                dataset_file = {}
                dataset_file["dataset_name"] = self.dataset_name
                dataset_file["labels_map"] = self.labels_map
                dataset_file["inverted_labels_map"] = self.inverted_labels_map                
                dataset_file["input_image_size"] = self.input_image_size
                dataset_file["output_image_size"] = self.output_image_size                
                dataset_file["train_proportion"] = self.train_proportion
                dataset_file["training_set"] = self.training_set
                dataset_file["validation_set"] = self.validation_set
                dataset_file["features_used"] = self.features_used
                
                with open(str(self.dataset_name)+".json", 'w') as f:
                        json.dump(dataset_file, f)

                print("==== WROTE "+str(self.dataset_name)+" =========")

        def createDataset(self, dataset_name, data_paths, input_image_size, output_image_size, features_used, train_Proportion):
                self.dataset_name = dataset_name
                self.input_image_size = input_image_size # our patches are square
                self.output_image_size = output_image_size # our patches are square
                self.data_paths = data_paths
                self.train_proportion = train_proportion
                self.current_batch = 0
                self.features_used = features_used
                self.buildLabelsMapAndInvertedLabelsMap(features_used)

                self.discarded_patchs = {}
                self.discarded_features = {}
                                
                self.training_set_paths = []                                
                self.validation_set_paths = []                
                self.validation_set_images = []
                self.validation_set_labels = []
                
                self.buildTrainingAndValidationSets()

                self.batches = []

                self.writeDataset()

        def fillDataset(self, dataset_file):
                dataset_json_file = open(dataset_file)
                data = json.load(dataset_json_file)


                self.dataset_name = data["dataset_name"]
                self.input_image_size = data["input_image_size"]
                self.output_image_size = data["output_image_size"]
                self.train_proportion = data["train_proportion"]
                self.labels_map = data["labels_map"]
                self.inverted_labels_map = data["inverted_labels_map"]
                self.training_set = data["training_set"]
                self.validation_set = data["validation_set"]
                self.features_used = data["features_used"]
                self.training_set_paths = []
                                             
                self.validation_set_paths = []                
                self.validation_set_images = []
                self.validation_set_labels = []

                for k, v in self.training_set.items():                        
                        self.training_set_paths.append(k)
                
                for k, v in self.validation_set.items():
                        im = self.readImage(k)
                        if im is None:
                                continue
                        
                        self.validation_set_paths.append(k)
                        self.validation_set_images.append(im)
                        self.validation_set_labels.append(v)
                
                
                self.current_batch = 0
                self.batches = []

                
if __name__ == "__main__":

        ds = Dataset()
        
        dataset_name = "dataset_name"
        
        data_path = []
        
        # Empty is always used, order is not important here
        features_used = [ "Ball"
                          ,"Empty"
                          ,"PostBase"
                          ,"Robot"
                          # ,"PenaltyMark"
                          ,"LineCorner"
                          # ,"ArenaCorner"
                          # ,"Center"
                          ,"T"
                          ,"X"
        ]

        input_image_size = 32
        output_image_size = 16
        train_proportion = 0.9
        
        ds.createDataset(dataset_name, data_path, input_image_size, output_image_size, features_used, train_proportion)        

        nb_batchs_in_epoch = ds.buildBatches(64)
        print("BATCHES SUCCESSFULLY BUILT")
        z, images, labels = ds.getBatch()
        print("getBatch() OK")
        aa, vv, cc = ds.getValidation()
        print("getValidation() OK")

        stats = ds.computeStatsOnSet(True)
        print("")
        print("")
        print("")
        for k, v in stats.items():
                print("")
                print(str(k)+" : ")
                for kk, vv in v.items():
                        if('_' in kk):
                                continue

                        mixed = 0
                        if(kk+"_in_mixed" in v):
                                mixed = v[kk+"_in_mixed"]
                        if kk != "mixed":
                                print('\t', "{0: <16}: {1: >8} ({2: >8} mixed)".format(kk, vv, mixed))
                        else:
                                print('\t', "{0: <16}: {1: >8}".format(kk, vv))

