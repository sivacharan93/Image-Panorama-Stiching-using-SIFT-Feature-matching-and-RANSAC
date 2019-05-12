#!/usr/bin/env python3

#importing required packages
import cv2
import numpy as np
import os, sys
import random
#import matplotlib.pyplot as plt
#from PIL import Image

#Part 1
## Hamming Distance of two descriptors
## Matching function to compute the number of matches
## Generate matrix function to create nxn matrix for similarity between image indices
## Clustering function called kmeans whose centre is the maximum matched image within the cluster

#Part 2
##Transform function
#Given a transformation matrix the transform function transforms the image coordinates to new coordinates and uses bilinear interpolation to fill in the pixel values
#Inverse warping and bilinear interpolation is used to avoid holes in the resulting image
#If the transformed coordinates are out of bounds in the original image, are left blank
##Solver function
#This function uses the pair of corresponding points from images and uses them 
#to compute for the transformation matrix
#A set of linear equations were solved based on the number of pair of points
#For translation, just a difference between the points would result in
# the translation vectors
#For Euclidian, affine and projective there are 4,6,8 equations to be solved
#The function only outputs a transformation matrix if the coefficients matrix
# of the linear equations is non singular
#Then it takes the transformation matrix and applies it to the input image 
#to transform it into the orientation of second image
#The resulting image is not entirely captured because the coordinates don’t
# fit the original image size

#Part 3
#Ransac  
#The interest points between two images are obtained from part 1 and they are 
#used to find the transformation matrix
#The points are then fed into RANSAC algorithm which selects the best hypothesis
# with maximum support i.e. inliers
#The RANSAC algorithm starts with fixed number of iterations (2500) and  high 
#inlier to total points ratio of 0.8. The Euclidian distance threshold is set to 5 pixels between transformed coordinates and original coordinates to count a point pair as an inlier. 4 points are used to find the hypothesis at a time.
#Further, it checks for best hypothesis with inlier ratio over 0.8, if no
# hypothesis is found then the inlier ratio is reduced by 0.05 and the hypothesis search is recursive until the best hypothesis is returned
#Panorama
#Using the best hypothesis returned from RANSAC i.e. the transformation 
#matrix the images are then stitched together to create a panorama
#For creating the panorama, a blank canvas of the size of both images 
#combined horizontally is created and then the first image is stored to the left
#The second image is then filled into the canvas using inverse warping and 
#bilinear interpolation using the inverse of the transformation matrix
#For overlapping locations, the pixel values are averaged

# =============================================================================
# part 1 functions
# =============================================================================

# This function is for reading image using image file address. Options are for color and grayscale.
def readim(address,option=1):
    if option == 1:
        img = cv2.imread(address, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(address)
    return img

# This function generates a maximum of 1000 ORB feature points from an image
def orbgen(img,n=1000):
    orb = cv2.ORB_create(nfeatures=n)
    #detect features
    (keypoints, descriptors) = orb.detectAndCompute(img, None)
    return keypoints,descriptors

# This function takes two descriptors and computes the hamming distance
def hamming(des1,des2):
    # des1 and des2 are arrays
    norm = cv2.norm( des1, des2, cv2.NORM_HAMMING)
    return norm
    
# We take one keypoint in one image and loop for all keypoints in other image
# We find closest and second closest matching descriptor distances and take their ratio with
# threshold = 0.8. If lesser than the threshold, we count it as a match    
def match(image1,image2,orb=orbgen,ratio_thresh=0.8):
    keypoints1,descriptors1 = orb(image1)
    keypoints2,descriptors2 = orb(image2)
    count = 0
    thresh1 = np.array(list(map(lambda x:x.response,keypoints1))).mean()
    thresh2 = np.array(list(map(lambda x:x.response,keypoints2))).mean()
    matkp1 = []
    matkp2 = []
    for i in range(len(keypoints1)):
        if keypoints1[i].response < thresh1:
            continue
        first = float('Inf')
        second = float('Inf')
        kp1 = 0
        kp2 = 0
        final = [first,second]
        for j in range(len(keypoints2)):
            if keypoints2[j].response < thresh2:
                continue
            distance = hamming(descriptors1[i],descriptors2[j])
            if min(final)>distance or max(final)>distance:
                final.remove(max(final))
                final.append(distance)
                kp1 = keypoints1[i]
                kp2 = keypoints2[j]
        
        first = min(final)
        second = max(final)
        ratio = first/second
    
        if ratio < ratio_thresh:
            count+=1
            matkp1.append(kp1)
            matkp2.append(kp2)
            
    return count,matkp1,matkp2

# This function is for putting the two images side by side on a large canvas and drawing lines
# matching the keypoints in both the images
def drawl(image1,image2,matkp1,matkp2):
    width = len(image1[0])+len(image2[0])
    height = max([len(image1),len(image2)])
# Change this to np.zeros((height,width,3)) for 3 channel image i.e RGB
    new = np.zeros((height,width))
    new[0:len(image1),0:len(image1[0])] = image1
    new[0:len(image2),len(image1[0]):len(new[0])] = image2
    for i,j in list(zip(matkp1,matkp2)):
        point1 = (int(i.pt[1]),int(i.pt[0]))
        point2 = (int(j.pt[1]),int(len(image1[0]) + j.pt[0]))

        cv2.line(new,(point1[1],point1[0]),(point2[1],point2[0]),(255,255,255),2)
        
    return new


# This function computes the no. of matches between two images based on index and assigns that
# value to both the i,j and j,i index to force symmetricity and reduce computation time
def genmat(images,matchf=match):
    new = np.zeros((len(images),len(images)))
    for i in range(len(images)):
        for j in range(i+1):
            new[i][j] = match(images[i],images[j])[0] 
            new[j][i] = new[i][j]
    return new


# This is the clustering function where we taken random indices and assign them as cluster centres.
# We loop for all indices except those which are already cluster centers and check for their nearest
# cluster and reassign it
# New cluster centers are calculated based on maximum matching value w.r.t all indices in same cluster
def kmeans(k,images,matrix,iter_limit=200):
    initial = random.sample(range(len(images)),k)
    centers = {i:initial[i-1] for i in range(0+1,k+1)}
    clusters = np.array([1 for i in range(len(images))])
    iterations = 0
    while iterations < iter_limit:
        index = 0
        for i in range(len(images)):
            if i in centers.values():
                continue
            score = -float('Inf')
            final_cluster = 1
            for c in range(k):
                cluster_centre = centers[c+1]
                new_score = matrix[index][cluster_centre]
                if new_score>score:
                    score = new_score
                    final_cluster = c+1
            
            clusters[index] = final_cluster
            index+=1
        
        iterations+=1
    
        # Here we change the cluster centres
        for c in range(1,k+1):
            indices = np.where(clusters==c)[0]
            sum1=0
            final_sum = 0
            final_ind = 0
            for ind in indices:
                new_list = list(indices)
                new_list.remove(ind)
                sum1 = np.sum(matrix[ind][new_list])
                if sum1>final_sum:
                    final_sum = sum1
                    final_ind = ind
                
            
            new_index = final_ind
            centers[c] = new_index
            
    return centers,clusters

#This function gives the accuracy score of the current clusters (PairWise Clustering Accuracy)
def accuracy_measure(clusters, files):
    n = len(clusters)
    TP = 0
    TN = 0

    for i in range(len(clusters)):
        for j in range(len(clusters)):
            if i!= j:
                name1 = files[i].strip().split('_')[0]
                name2 = files[j].strip().split('_')[0]
                if name1 == name2:
                    if clusters[i] == clusters[j]:
                        TP += 1
                else:
                    if clusters[i] != clusters[j]:
                        TN += 1
    
    print('True positive is {}'.format(TP))
    print('True negative is {}'.format(TN))
    print('Accuracy is : {}'.format((TP+TN)/(n*(n-1))))
    acc=  (TP+TN)/(n*(n-1))
    return acc


# =============================================================================
# part 2 functions
# =============================================================================
#Neighbor function to return the sorrounding points of a coordinate point
def neighbors(x,y,matrix):
    new = matrix.dot([y,x,1])
    x_floor,y_floor,x_ceil,y_ceil = int(np.floor(new[1]/new[2])),int(np.floor(new[0]/new[2])),\
                                    int(np.ceil(new[1]/new[2])),int(np.ceil(new[0]/new[2]))
    return (new[1]/new[2])-x_floor,(new[0]/new[2])-y_floor,x_floor,y_floor,x_ceil,y_ceil

#Function to transform the image using the transformation matrix
def transform(image,matrix):
    l,d,c = image.shape[0],image.shape[1],image.shape[2]
    inv_matrix = np.linalg.inv(matrix)
    new_image = np.zeros((l,d,c))
    for i in range(l):
        for j in range(d):
            x,y,x_floor,y_floor,x_ceil,y_ceil = neighbors(i,j,inv_matrix) 
            if all(c in range(l) for c in [x_floor,x_ceil])  and \
            all(c in range(d) for c in [y_floor,y_ceil]) :
                    new_image[i,j] = (1-x)*(1-y)*image[x_floor,y_floor] + \
                                     (1-y)*(x)*image[x_ceil,y_floor] + \
                                     (y)*(1-x)*image[x_floor,y_ceil] + \
                                     (y)*(x)*image[x_ceil,y_ceil] 
    return new_image

##Solver function to solve for the transformation matrix using the corresponding points
#across images
# # We have looked into the follwing URL's for formulating the linear equations
# #https://math.stackexchange.com/questions/77462/finding-transformation-matrix-between-two-2d-coordinate-frames-pixel-plane-to
# https://stackoverflow.com/questions/11687281/transformation-between-two-set-of-points
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node11.html
def solver(n,pre,post):
    if n ==1:
        trans = np.array(post) - np.array(pre)
        return [[1,0,trans[0]],[0,1,trans[1]],[0,0,1]]
    if n == 2:
        matrix = [[pre[0],-pre[1],1,0],
                  [pre[1], pre[0],0,1],
                  [pre[2],-pre[3],1,0],
                  [pre[3], pre[2],0,1]]
        
        values = np.linalg.solve(matrix, post)
        values = [[values[0],-values[1],values[2]],[values[1],values[0],values[3]],[0,0,1]]
        return values
    if n == 3:
        matrix = [[pre[0], pre[1], 1, 0, 0, 0],
                  [0, 0, 0, pre[0], pre[1], 1],
                  [pre[2], pre[3], 1, 0, 0, 0],
                  [0, 0, 0, pre[2], pre[3], 1],
                  [pre[4], pre[5], 1, 0, 0, 0],
                  [0, 0, 0, pre[4], pre[5], 1]]
        
        values = np.linalg.solve(matrix, post)
        values = np.vstack((values.reshape(2,3),np.array([0,0,1])))
        return values
    if n==4:
        matrix = [[pre[0], pre[1], 1, 0, 0, 0,-post[0]*pre[0],-post[0]*pre[1]],
                  [0, 0, 0, pre[0], pre[1], 1,-post[1]*pre[0],-post[1]*pre[1]],
                  [pre[2], pre[3], 1, 0, 0, 0,-post[2]*pre[2],-post[2]*pre[3]],
                  [0, 0, 0, pre[2], pre[3], 1,-post[3]*pre[2],-post[3]*pre[3]],
                  [pre[4], pre[5], 1, 0, 0, 0,-post[4]*pre[4],-post[4]*pre[5]],
                  [0, 0, 0, pre[4], pre[5], 1,-post[5]*pre[4],-post[5]*pre[5]],
                  [pre[6], pre[7], 1, 0, 0, 0,-post[6]*pre[6],-post[6]*pre[7]],
                  [0, 0, 0, pre[6], pre[7], 1,-post[7]*pre[6],-post[7]*pre[7]]]
        values = np.linalg.solve(matrix, post)
        
        values = [[values[0],values[1],values[2]],[values[3],values[4],values[5]],[values[6],values[7],1]]
        return values

# =============================================================================
# part 3 functions
# =============================================================================
#Ransac algorithm for finding the best hypothesis using the match 
#points across two images
# We have looked into the follwing URL for implementing RANSAC
# https://stackoverflow.com/questions/4655334/ransac-algorithm
def ransac(img_1_points,img_2_points,iterations,threshold,ratio,n):
    #print ("ransac called")
    model_out = np.array([])
    #error_out = np.Inf
    if len(img_1_points) < 4:
        print("Doesnt have enough match points for RANSAC")
        exit  
    count = 0
    v_ratio = 0
    while(count<=iterations):
        count += 1        
        random_indices = np.array(np.random.randint(0,len(img_1_points)-1,4))
        pre = list(img_1_points[random_indices].flatten())
        post = list(img_2_points[random_indices].flatten())
        #get the corresponding matrix   
        #print(pre,post)
        try:
            matrix = np.array(solver(n,pre,post))
        except:
            continue
        #Voting for consensus
        votes = 0
        tot_error = 0
        
        for j in range(len(img_1_points)):
            x = img_1_points[j]
            u = img_2_points[j]            
            u_t = matrix.dot(np.array([x[0],x[1],1]))            
            u_t = np.array([u_t[0]/u_t[2],u_t[1]/u_t[2]])
            #print(x,u,u_t)
            dst = np.linalg.norm(u_t-u)
            
            if dst < threshold:
                votes += 1
                tot_error += dst
        #if votes/len(img_1_points) > ratio and tot_error < error_out:
        
        if v_ratio < votes/len(img_1_points):
            v_ratio = votes/len(img_1_points)
            #print(v_ratio)
            #print(matrix,v_ratio)
            model_out = matrix
            #error_out = tot_error
            
    if v_ratio > ratio :        
        return model_out
    else:
        return ransac(img_1_points,img_2_points,iterations,threshold,ratio-0.05,n)

#Function for stiching two images together using the transformation matrix
def panorama(image1,image2,matrix,neighbors1=neighbors):
    
    width = len(image1[0])+int(0.5*len(image2[0]))
    height = max([len(image1),len(image2)])
    new = np.zeros((height,width,3))
    inv_matrix = np.linalg.inv(matrix)
    
    new[0:len(image1),0:len(image1[0])] = image1

    ylim = len(image2[0])
    xlim = len(image2)
    
    for j in range(len(new[0])):
        for i in range(len(new)):
            x,y,x_floor,y_floor,x_ceil,y_ceil = neighbors1(i,j,inv_matrix)
    #        print(i,j)
    #        print()
            pixel1 = new[i][j]
            if (x_ceil>=0 and x_ceil<xlim and x_floor>=0 and x_floor<xlim and y_ceil>=0 and y_ceil<ylim and y_floor>=0 and y_floor<ylim ):
    #            print(i,j)        
                pixel2 = (1-x)*(1-y)*image2[x_floor,y_floor] + \
                                         (1-y)*(x)*image2[x_ceil,y_floor] + \
                                         (y)*(1-x)*image2[x_floor,y_ceil] + \
                                         (y)*(x)*image2[x_ceil,y_ceil] 
    #        
            
                if int(pixel1.sum()):
                    pixel = (pixel1+pixel2)/2
                    new[i,j] = pixel.astype('int')
                else:
                    new[i,j] = pixel2
    idx = np.argwhere(np.all(new[..., :] == 0, axis=0))                
    new = np.delete(new, idx, axis=1)
    return new

            
#Main function
if __name__ == '__main__':
    print ("main function called")
    part = sys.argv[1]
    if part == "part1":
        print ("part1 function")
        k = int(sys.argv[2])
        image_list = sys.argv[3:]
        out_file = image_list.pop()
#        rest sys.arv but lSt oe image list
#        last argv outfile
#        image_list = os.listdir(str(os.getcwd())+'/'+'part1-images')
#        images = []
        files = image_list
        
        image_numpy_list = [readim(i) for i in image_list]
            
        matrix = genmat(image_numpy_list)
        mat = matrix.copy()
        
        centers,clusters = kmeans(k,image_numpy_list,mat,700)
        clustersnp = np.array(clusters)
        
        with open(out_file, "w") as f:
            for i in range(1,k+1):
                cluster = i
                indices = np.where(clustersnp==i)[0]
                ff = [files[ind] for ind in indices]
                print(ff)
                write = ' '.join(ff)
                f.write(write+"\n")
                
        f.close()
#        accuracy = accuracy_measure(clusters, files)
#        print ("Current Accuracy: " + str(accuracy))
        print ("part 1 ended, saved outfile")

    elif part == "part2":
        print ("part2 function")
#        n = sys.argv[2]
#        input_im = readim(file,0)
#        new_im = transform(input_im,matrix)    
#        Image.fromarray(new_im.astype('uint8')).show()
#        cv2.imwrite(output,new_im.astype('uint8'))
        n,img_1,img_2,img_output = int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]
        input_im = readim(img_1,0)
        #Store the coordinate points input
        pre = list(map(int,sum([sys.argv[6:][i:i + 2] for i in range(0, len(sys.argv[6:]), 2) if (i/2)%2 == 0],[])))
        post = list(map(int,sum([sys.argv[6:][i:i + 2] for i in range(0, len(sys.argv[6:]), 2) if (i/2)%2 != 0],[])))
        #Solve for transformation matrix
        matrix = solver(n,pre,post)
        #print the matrix to console
        print(matrix)
        #Trasnform the image using the matrix obtained from solver
        new_im=transform(input_im,matrix)
        #Save the output
        cv2.imwrite(img_output,new_im.astype('uint8'))
        print ("part2 ended, saved output image")

    elif part == "part3":
        print ("part3 function")
        #RANSAC parameters
        iterations = 2500  # ransac iterations
        threshold = 5  #threshold for error
        ratio = 0.8 # inliers to outliers
        n = 4 #no of points to find the tranform matrix
        
        image_1,image_2,output = sys.argv[2],sys.argv[3],sys.argv[4]
        
        #reading greyscale images
        image1 = readim(image_1)
        image2 = readim(image_2)
        
        #get the matching points
        count,match1,match2 = match(image1,image2)
        side = drawl(image1,image2,match1,match2)
        
        points1 = np.array([i.pt for i in match1])
        points2 = np.array([i.pt for i in match2])
        
        #read rgb images
        image1 = readim(image_1,0)
        image2 = readim(image_2,0)   
        
        
        #get the transform matrix using ransac
        transform_matrix = ransac(points2,points1,iterations,threshold,ratio,n)
        #Stich panorama using the transformation matrix and original images
        output_image = panorama(image1,image2,transform_matrix)
        
        #Saving the output to drive
        cv2.imwrite(output,output_image)
        print ("part3 ended, saved output image")

    else:
        print ("please enter valid parameters")
        sys.exit()
