
from IPython import display
display.clear_output()
import numpy as np
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import torch
from IPython.display import display, Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import collections
import os
import utils 
import torch.nn as nn
import torchvision


def minibatch(*tensors, **kwargs):

    batch_size = kwargs['batch_size']

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
# Configuration settings
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 3
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    batch_size = 32
    criterion = nn.CrossEntropyLoss()

class Preprocess_yolo:
    def __init__(self,model_path = "best2.pt"):
        self.model = YOLO(model_path)

    def preprocess_yolo(self,results):
        contours = []
        upper_mask = np.zeros((224,224), np.uint8)
        lower_mask = np.zeros((224,224), np.uint8)
        b_mask = np.zeros((224,224), np.uint8)
        for r in results:
            
            img = np.copy(r.orig_img)
            img_name = Path(r.path).stem  # source image base-name
            
            for ci, c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]
                
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                contours.append(contour)
                _ = cv2.drawContours(b_mask, [contours[-1]], -1, 255, 1)
                mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
                # res.append(mask3ch)
                isolated = cv2.bitwise_and(mask3ch, img)



            _ = cv2.drawContours(upper_mask, [contours[0]], -1, 255, 1)
            _ = cv2.drawContours(lower_mask, [contours[1]], -1, 255, 1)
            
        nonzero = b_mask.nonzero() 
        nonzero_upper = upper_mask.nonzero() 
        nonzero_lower = lower_mask.nonzero()   
        upper_point = []
        lower_point = []
        point =[]
        for i in range(nonzero_upper[1].shape[0]):
            upper_point.append([nonzero_upper[1][i], nonzero_upper[0][i]])
        for i in range(nonzero_lower[1].shape[0]):
            lower_point.append([nonzero_lower[1][i], nonzero_lower[0][i]])
        for i in range(nonzero[1].shape[0]):
            point.append([nonzero[1][i], nonzero[0][i]])


        upper_point = np.array(upper_point)
        lower_point = np.array(lower_point)
        point  = np.array(point)
        upper_point = np.array([p for p in list(upper_point) if p[1]!=1])
        lower_point = np.array([p for p in list(lower_point) if p[1]!=223])
        return mask3ch,point,upper_point,lower_point  
    
    
    def get_4_contour_point(self, point,upper_point,lower_point):
    
        x_min_upper = np.min(upper_point[:,0])
        x_max_upper = np.max(upper_point[:,0])
        x_min_lower =np.min(lower_point[:,0])
        x_max_lower =   np.max(lower_point[:,0])

        y_min_upper = np.min(upper_point[:,1])
        y_max_upper = np.max(upper_point[:,1])
        y_min_lower =np.min(lower_point[:,1])
        y_max_lower =   np.max(lower_point[:,1])

        x_min = np.max([x_min_upper,x_min_lower])
        x_max = np.min([x_max_upper,x_max_lower])
        # p_intersect = []
        # for p in upper_point:
        #     if p[1] == y_min_lower:
        #         if  not p_intersect: 
        #             p_intersect.append(p)
        #         else:
        #             if abs(p[0] - p_intersect[-1][0])>=10:
        #                 p_intersect.append(p)
        # x_min_left = max(p_intersect[0][0],x_min_upper,x_min_lower)
        # x_max_left= p_intersect[1][0]
        # x_min_right= p_intersect[2][0]
        # x_max_right = min(p_intersect[3][0],x_max_upper,x_max_lower)

        # return  x_min_left,x_max_left,x_min_right,x_max_right
        return x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max
    def get_unique_pixel_upper(self,lst):
        new_lst = []
        unique_set = set()
        for ele in reversed(lst):
            if ele[0] not in unique_set:
                unique_set.add(ele[0])
                new_lst.append(ele)
        return sorted(new_lst, key=lambda x: x[0])

    def get_unique_pixel_lower(self,lst):
        new_lst = []
        unique_set = set()
        for ele in lst:
            if ele[0] not in unique_set:
                unique_set.add(ele[0])
                new_lst.append(ele)
        return sorted(new_lst, key=lambda x: x[0])
    
    def get_4_set_point(self,upper_point,lower_point,x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max):
        upper_point = self.get_unique_pixel_upper(upper_point)
        lower_point = self.get_unique_pixel_lower(lower_point)
        upper_left = []
        upper_right = []
        lower_left = []
        lower_right = []

        mid_point = int((x_min_upper+x_max_upper+x_min_lower+x_max_lower)/4)
        l = int((x_max_upper-x_min_upper+ x_max_lower-x_min_lower)/4)
        r =0.2
        remove_range = [i for i in range(int(mid_point-r*l/2),int(mid_point+r*l/2))]
        left = int(mid_point-r*l/2)
        right = int(mid_point+r*l/2)
        for point in upper_point:
            if point[0] not in remove_range:
                if point[0]<= left  and point[0]>=x_min+l//8:
                    upper_left.append(point)
                if point[0] >=right and point[0]<=x_max-l//8:
                    upper_right.append(point)

        for point in lower_point:
            if point[0] not in remove_range:
                if point[0]<= left  and point[0]>=x_min+l//8:
                    lower_left.append(point)
                if point[0] >=right and point[0]<=x_max-l//8:
                    lower_right.append(point)
        # upper_right = get_unique_pixel(upper_right)
        # upper_left= get_unique_pixel(upper_left)
        # lower_right = get_unique_pixel(lower_right)
        # lower_left = get_unique_pixel(lower_left)
        return upper_left,upper_right,lower_left,lower_right
    def visual_result(self,img_path = "9003175L.png"):
        with torch.no_grad():
            results = self.model.predict(task="segment", source=img_path, conf=0.25, hide_labels=True, show_conf=False, save=False, augment=False, boxes=False) 
        mask3ch,point,upper_point,lower_point   = self.preprocess_yolo(results)
        x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max=self.get_4_contour_point( point,upper_point,lower_point)
        upper_left,upper_right,lower_left,lower_right = self.get_4_set_point(upper_point,lower_point,x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max) 
        test = mask3ch.copy()
        for i in range(0,len(upper_left),3): 
            cv2.line(test,upper_left[i],lower_left[i],255,1)
        for i in range(0,len(upper_right),3): 
            cv2.line(test,upper_right[i],lower_right[i],255,1)
        plt.imshow(test)

    def pipeline(self,img_path = "9003175L.png"):
        with torch.no_grad():
            results = self.model.predict(task="segment", source=img_path, conf=0.25, hide_labels=True, show_conf=False, save=False, augment=False, boxes=False) 
        mask3ch,point,upper_point,lower_point   = self.preprocess_yolo(results)
        x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max=self.get_4_contour_point( point,upper_point,lower_point)
        upper_left,upper_right,lower_left,lower_right = self.get_4_set_point(upper_point,lower_point,x_min_upper,x_max_upper,x_min_lower,x_max_lower,x_min,x_max) 
        left,right =self.calculate_distance(upper_left,upper_right,lower_left,lower_right)
        segment_left,segment_right = self.divide_2_10_segment(left,right)
        result = self.jsw_processing(segment_left,segment_right)
        return result
    def calculate_distance(self,upper_left,upper_right,lower_left,lower_right):
        right = np.array([abs(upper_right[i][1]-lower_right[i][1]) for i in range(len(upper_right))])
        left = np.array([abs(upper_left[i][1]-lower_left[i][1]) for i in range(len(upper_left))])
        return left,right 
    def divide_2_10_segment(self,left,right):
        idx_left = np.linspace(0,len(left)-1,num=11,dtype = int)
        idx_right = np.linspace(0,len(right)-1,num=11,dtype = int)
        segment_left =[]
        segment_right = []
        for i in range(9):
            # segment_left.append(np.mean(idx_left[i:i+1]))
            # segment_right.append(np.mean(idx_right[i:i+1]))
            segment_left.append(np.mean(left[idx_left[i]:idx_left[i+1]]))
            segment_right.append(np.mean(right[idx_right[i]:idx_right[i+1]]))
        segment_left.append(np.mean(left[idx_left[9]:]))
        segment_right.append(np.mean(right[idx_right[9]:]))
        return np.array(segment_left),np.array(segment_right)
    def jsw_processing(self,segment_left,segment_right):
        jsw_max = np.max(np.concatenate((segment_left,segment_right)))
        jsw_min = np.min(segment_left) if jsw_max in segment_right else np.min(segment_right)
        jsw_mean = (abs(np.mean(segment_left)-np.mean(segment_right)))/jsw_max 
        jsw_mm = (jsw_max - jsw_min )/jsw_max
        return np.array([jsw_mean,jsw_mm])
    


class Preprocess_GD:
    def __init__(self,root_path = "data"):
        self.path = root_path
    def pipeline(self):
        new_label={
        0:0,
        1:1,
        2:1,
        3:2,
        4:2
            }
        PATH = self.path+"/train/"
        xdata = collections.defaultdict(list)
        ytrain = []
        for classes in [0,1,2,3,4]:
            ls =  os.listdir(PATH+str(classes))
            print(f"Processing images class: {classes}")
            for samples in ls:
                img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(224,224))
                img = img.transpose((2, 0, 1))
                xdata[new_label[classes]].append(img)
                ytrain.append(new_label[classes])
        
        PATH = self.path+"/test/"
        tdata = collections.defaultdict(list)
        ytest = []
        for classes in [0,1,2,3,4]:
            ls =  os.listdir(PATH+str(classes))
            print(f"Processing images class: {classes}")
            for samples in ls:
                img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(224,224))
                img = img.transpose((2, 0, 1))
                tdata[new_label[classes]].append(img)
                ytrain.append(new_label[classes])
        PATH = self.path+"/val/"
        vdata = collections.defaultdict(list)
        yval = []
        for classes in [0,1,2,3,4]:
            ls =  os.listdir(PATH+str(classes))
            print(f"Processing images class: {classes}")
            for samples in ls:
                img = cv2.resize(cv2.imread(PATH+str(classes)+'/'+samples),(224,224))
                img = img.transpose((2, 0, 1))
                vdata[new_label[classes]].append(img)
                yval.append(new_label[classes])
            
        Xtrain = xdata[0] + xdata[1] + xdata[2]
        Ytrain = [0 for i in range(len(xdata[0]))] + [1 for i in range(len(xdata[1]))] + [2 for i in range(len(xdata[2]))]
        Xtest = tdata[0] + tdata[1] + tdata[2]
        Ytest = [0 for i in range(len(tdata[0]))] + [1 for i in range(len(tdata[1]))] + [2 for i in range(len(tdata[2]))]
        Xval = vdata[0] + vdata[1] + vdata[2]
        Yval = [0 for i in range(len(vdata[0]))] + [1 for i in range(len(vdata[1]))] + [2 for i in range(len(vdata[2]))]

        Xtrain =torch.tensor(np.array(Xtrain),dtype=torch.float32)
        Ytrain =torch.tensor(np.array(Ytrain),dtype=torch.long)
        Xtest = torch.tensor(np.array(Xtest),dtype=torch.float32)
        Ytest = torch.tensor(np.array(Ytest),dtype=torch.long)
        Xval = torch.tensor(np.array(Xval),dtype=torch.float32)
        Yval = torch.tensor(np.array(Yval),dtype=torch.long)

        return Xtrain,Ytrain,Xtest,Ytest,Xval,Yval

class Preprocess_RF:
    def __init__(self):
        self.config = Config()
        self.model = self.config.model
        self.model.fc = nn.Linear(self.model.fc.in_features, 3)
        self.model.load_state_dict(torch.load("/Users/namle/Desktop/SegJSW/bestresnet.pt",map_location=torch.device('cpu')))

        self.process_yolo = Preprocess_yolo()
    def process(self,Xset):
        result = torch.tensor([])
        for (minibatch_num,(Xbatch)) in enumerate(minibatch(Xset,batch_size=32)):
            resnet = self.model(Xbatch)
            yolo = torch.tensor([])
            for i in range(Xbatch.shape[0]):
                res = torch.tensor(self.process_yolo.pipeline(Xbatch[i])).unsqueeze(0)
                yolo = torch.concat((yolo,res),dim = 0)
            resbatch = torch.concat((resnet,yolo),dim =-1)
            result = torch.concat((result,resbatch),dim=0)
        return result 

                




            


