"""
Derived from: https://github.com/kratzert/finetune_alexnet_with_tensorflow/
"""

import numpy as np
import cv2

class BatchPreprocessor(object):
# shuffle 拖曳
    def __init__(self,dataset_file_path,num_classes,output_size=[224,224],horizontal_flip=False,shuffle=False,mean_color=[132.2766,139.6506,146.9702],multi_scale=None):
        self.num_classes=num_classes
        self.output_size=output_size
        self.horizontal_flip=horizontal_flip
        self.shuffle=shuffle
        self.mean_color=mean_color
        self.multi_scale=multi_scale

        self.pointer=0
        self.images=[]
        self.labels=[]

        # Read the dataset file
        dataset_file=open(dataset_file_path)
        # lines=dataset_file.readlines()
        for line in dataset_file.readlines():
            items=line.split()# The items be like :temp/images/image.jpg 0(The last number is labels number)
            self.images.append(items[0])
            self.labels.append(int(items[1]))

        # Shuffle the data 
        if self.shuffle:
            self.shuffle_data()

    def shuffle_data(self):
        images=self.images[:]
        labels=self.labels[:]
        self.images=[]
        self.labels=[]

        # use the np.random.permutation to random the image file index
        idx=np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.pointer=0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self,batch_size):
        # Get next batch of image (path) and labels
        paths=self.images[self.pointer:(self.pointer+batch_size)]
        labels=self.labels[self.pointer:(self.pointer+batch_size)]

        # Update pointer
        self.pointer=self.pointer+batch_size

        # Read images(RGB)
        images=np.ndarray([batch_size,self.output_size[0],self.output_size[1],3])
        for i in range(len(paths)):
            img=cv2.imread(paths[i])

            # Flip image at random if flag is selected
            if self.horizontal_flip and np.random.random()<0.5:
                img=cv2.flip(img,1)

            if self.multi_scale is None:
                # Resize image for output
                img=cv2.resize(img,(self.output_size[0],self.output_size[0]))
                img=img.astype(np.float32)
            elif isinstance(self.multi_scale,list):# This function smilar to type ,but this function can check the whether it is subclass or not
                # Resize image for random scale
                new_size=np.random.randint(self.multi_scale[0],self.multi_scale[1],1)[0]# The randint final parameter is mean the size of the random number ,ex:np.random.randint(0,5,1)=4,and the [0] it mean take the np.array out to become the int 
                img =cv2.resize(img,(new_size,new_size))
                img=img.astype(np.float32)

                # random crop at output size
                diff_size=new_size-self.output_size[0]
                random_offset_x=np.random.randint(0,diff_size,1)[0]
                random_offset_y=np.random.randint(0,diff_size,1)[0]
                img=img[random_offset_x:(random_offset_x+self.output_size[0]),random_offset_y:(random_offset_y+self.output_size)]

            # Subtract mean color
            img=img-np.array(self.mean_color)

            images[i]=img

        # Expand labels to one hot encoding
        one_hot_labels=np.zeros((batch_size,self.num_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]]=1

        # Return array of images and labels
        return images,one_hot_labels

