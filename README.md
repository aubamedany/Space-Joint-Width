**best.pt**to store pretrianed YOLO model which was used to do segmentation

**bestresnet.pt** to store model classification.

Data was preprocessed in preprocess.py

The flow of processing is descriped in the flow diagram:
![alt text](image-1.png)

Using **preprocess_GD** function to get preprocess vecto of classifier module (dim=3)

Using **preprocess_YOLO** function to get preprocess vecto of segmentation module (dim =2)

Image after using preprocess_YOLO function
![image](https://github.com/aubamedany/Space-Joint-Width/assets/129536301/8b4975d7-3dff-4947-b5fd-2f076d6fecdf)

Using **preprocess_RF** function to get concatenated vecto for input to final classifier model.
