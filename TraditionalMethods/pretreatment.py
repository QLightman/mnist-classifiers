#/usr/bin/python3
#This code was meant to operate noise reduction and size adjustment
import cv2
import numpy as np
import argparse
import sys
import random
from PIL import Image
class Array_class:
    """
    this class was meant to save the connected component and its size
    """
    def __init__(self, number, array):
        self.number = number
        self.array = array

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default="mnist_train_data", type=str, help='path of the data')
parser.add_argument('--origin_height', default=45, type=int,help='height of each image')
parser.add_argument('--origin_width', default=45, type=int,help='width of each image')
parser.add_argument('--height', default=20, type=int,help='height of each image you want')
parser.add_argument('--width', default=20, type=int,help='width of each image you want')
parser.add_argument('--size', default=100, type=int,help='number of images to be transformed')
parser.add_argument('--save_images', default=0, type=int,help='whether to save the image or not')
parser.add_argument('--ratio', default=0.45, type=float,help='decides whether to add the second largest connected component and so on')
parser.add_argument('--save_file', default=0, type=int,help='whether to save all the image file into a npy file or not')
parser.add_argument('--hog', default=0, type=int,help='whether to use  Histogram of Oriented Gradients or not')
parser.add_argument('--normalize', default=0, type=int,help='whether to normalize the data or not')
parser.add_argument('--rotate', default=0, type=int,help='whether to rotate the data or not')


global_index=0
index_list=np.array([0,0])

def pretreatment(file_path="mnist_train_data",origin_height=45,origin_width=45,
                 height=20,width=20,size=100,save_images=0,ratio=0.45,save_file=0,rotate=0,normalize=0,hog=0):
    data = np.fromfile(file_path,dtype=np.uint8)
    data = data.reshape(-1,45,45)
    data=data[:size]
    if normalize==1:
        data=np.where(data>128,255,0)
    if save_images==1:
        if rotate==1:
            string="rotate_before_change"
        else:
            string="before_change"
        for i in range(data.shape[0]):
            im = Image.fromarray(np.uint8(data[i]))
            im.save(str(i)+string+".png")
    if rotate==1:
        for index,value in enumerate(data):
            new_im = Image.fromarray(value.astype(np.uint8))
            new_im = new_im.rotate(random.randint(-15, +15))
            data[index]=(np.matrix(new_im.getdata(),dtype=np.uint8).reshape((origin_height,origin_width)))
    #repalce_data is to save the pretreated image
    replace_data=np.zeros([data.shape[0],height,width])
    if hog==1:
        hog_data=np.zeros([data.shape[0],18,18])
    #hog_data is to save the image processed by HOG
    print("after", data.shape)
    global index_list
    for index,value  in enumerate(data):
        test_data=np.zeros([origin_height,origin_width])
        Operator_list=[]
        for i in range(origin_height):
            for j in range(origin_width):
                if value[i][j]>0 and test_data[i][j]==0:
                    global global_index
                    global_index=0
                    index_list=np.array([0,0])
                    interation(value,i,j,test_data,origin_height,origin_width)
                    index_list=np.delete(index_list,0,axis=0)
                    Operator_list.append(Array_class(index_list.shape[0],index_list))

        data[index]=detect_leave(sorted(Operator_list, key=lambda a: a.number),value,ratio)

    #cutting the margin of each image
    bottom=0
    right=0
    top=0
    left=0
    bottom_top_margin=np.array([])
    right_left_margin=np.array([])
    for index,value  in enumerate(data):
        for j in range(origin_height):
            if value[j].sum()!=0:
                top=j
                break
        for j in range(origin_height-1,-1,-1):
            if value[j].sum()!=0:
                bottom=j
                break
        for j in range(origin_width):
            if value[:,j].sum()!=0:
                left=j
                break
        for j in range(origin_width-1,-1,-1):
            if value[:,j].sum()!=0:
                right=j
                break
        bottom_top_margin=np.append(bottom_top_margin,bottom-top)
        right_left_margin=np.append(right_left_margin,right-left)
        tmp=np.zeros([bottom-top,right-left])
        for i in range(top,bottom):
            for k in range(left,right):
                if i>=origin_height or k>=origin_width:
                    continue
                tmp[i-top][k-left]=value[i][k]
        tmp=add_margin(tmp)
        im = Image.fromarray(np.uint8(tmp))
        im=im.resize((height,width))
        if save_images==1:
            if rotate==1:
                string="rotate_after_change"
            else:
                string="after_change"
            im.save(str(index)+string+".png")
        replace_data[index]=(np.matrix(im.getdata(),dtype=np.uint8).reshape((height,width)))
    # print ("margin test", bottom_top_margin.shape)
    # print ("max_bottom_top",bottom_top_margin.max())
    # print ("max_bottom_top_median",np.median(bottom_top_margin))
    # print ("margin test", right_left_margin.shape)
    # print ("max_right_left",right_left_margin.max())
    # print ("max_right_left_median",np.median(right_left_margin))


    if hog==1:
        for index,value in enumerate(replace_data):
            hog = cv2.HOGDescriptor('hog.xml')
            img = np.reshape(value,(height,width))
            cv_img = img.astype(np.uint8)
            hog_data[index] = hog.compute(cv_img).reshape(18,18)
        replace_data=hog_data
    if save_file==1:
        if rotate==1:
            string="rotate_"
        else:
            string="_"
        np.save(str(size)+string+'.npy',replace_data)

    return replace_data


def detect_leave(List,data,ratio):
    """
    This function used to reduced the noise of the original image
    :param List: List is an Array_class, which contians the connected components and its size
    :param data: data is a single image
    :param ratio: ratio is to determine how many connected components to leave
    :return: modified image, which had reduced noise, leaving only the largest connected components
    if the size of over 45%
    """
    List.reverse()
    num=data.sum()/255
    tmp=0
    triping_index=0
    for index,value in enumerate (List):
        tmp=tmp+value.number
        if tmp*1.0/num>ratio:
            triping_index=index
            break
    for i in range(triping_index+1,len(List)):
        for point in  List[i].array:
            data[point[0],point[1]]=0
    return data


def add_margin(data):
    """
    This function used to make each image a square,so the shape of each figure can be saved.
    :param data: data is a single image
    :return: image which had been added margin to become a square
    """
    length=np.max([data.shape[0],data.shape[1]])
    result=np.zeros([length,length])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[(length-data.shape[0])/2+i][(length-data.shape[1])/2+j]=data[i][j]
    return result

def interation(data,indexi,indexj,test_data,origin_height,origin_width):
    """
    This function use a recurrence method to detect the connected components of each image.
    :param data: data is a single image
    :param indexi: the current row the algorithm had detected.
    :param indexj:the current column the algorithm had detected.
    :param test_data: the data to detect the connected components of each image,
     so that the original image won't change
    :param origin_height: the height of the original image
    :param origin_width: the width of the original image
    """
    if(indexi<0 or indexi>=origin_height):
        return
    if(indexj<0 or indexj>=origin_width):
        return
    if(data[indexi][indexj]==0):
        return
    if(test_data[indexi][indexj]==1):
        return
    global global_index
    global_index=global_index+1

    test_data[indexi][indexj]=1
    global index_list
    index_list=np.vstack((index_list,[indexi,indexj]))
    next_list=[[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
    for i in next_list:
        interation(data,indexi+i[0],indexj+i[1],test_data,origin_height,origin_width)


def main():
    args = parser.parse_args(sys.argv[1:])
    data = np.fromfile(args.file_path,dtype=np.uint8)
    data = data.reshape(-1,args.origin_height,args.origin_width)
    return pretreatment(file_path=args.file_path,origin_height=args.origin_height,origin_width=args.origin_width,
                 height=args.height,width=args.width,size=args.size,save_images=args.save_images,
                        ratio=args.ratio,save_file=args.save_file,rotate=args.rotate, hog=args.hog,normalize=args.normalize)

if __name__ == "__main__":
    main()
