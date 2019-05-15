# Design decisions, simplifications and assumptions:
# Part 1: Image matching and clustering
    ●	We used OpenCV implementation of ORB algorithm for feature keypoint detection.
    ●	Bitwise Hamming distance was used as the distance between the descriptor features.
    ●	Match method was implemented that counts the number of matching descriptors for keypoints based on threshold between first closest          and second closest matches.
    ●	The chamfer distance like matching method above is not symmetric.
    ●	Response attribute of the keypoint object was used to filter out the large number of keypoints in each image.
    ●	Keypoints whose response attribute value was less than the mean of all response attribute values in an image were discarded to make         computation faster.
    ●	The threshold was set at 0.8 to get optimal matching.
    ●	A similarity matrix is constructed for the images to be clustered. Only the bottom half of the diagonal is filled and is copied over      to the top half to reduce computation time.
    ●	The clustering algorithm implemented is a variant of K-means where we take the cluster centroid as the image with the maximum matches       within it’s cluster.
    ●	Clustering algorithm uses the similarity matrix and image indices to perform clustering.
    ●	Clustering algorithm recenters the cluster centre based on the maximum number of matches with the other images in the same cluster.
    ●	The number of iterations was set to 700.
 # Part 2: Image Transformations
    1.	Transforming an image using transformation matrix
      Design and specifications
        •	Given a transformation matrix the transform function transforms the image coordinates to new coordinates and uses bilinear                interpolation to fill in the pixel values
        •	Inverse warping and bilinear interpolation is used to avoid holes in the resulting image
        •	If the transformed coordinates are out of bounds in the original image, are left blank

    2.	Finding the transformation matrix from points and applying it to input image
      Design, specifications and assumptions
        •	This function uses the pair of corresponding points from images and uses them to compute for the transformation matrix
        •	A set of linear equations were solved based on the number of pair of points
        •	For translation, just a difference between the points would result in the translation vectors
        •	For Euclidian, affine and projective there are 4,6,8 equations to be solved
        •	The function only outputs a transformation matrix if the coefficients matrix of the linear equations is non singular
        •	Then it takes the transformation matrix and applies it to the input image to transform it into the orientation of second image
        •	The resulting image is not entirely captured because the coordinates don’t fit the original image size

# Part 3: Image matching and transformations
    •	The interest points between two images are obtained from part 1 and they are used to find the transformation matrix
    •	The points are then fed into RANSAC algorithm which selects the best hypothesis with maximum support i.e. inliers
    •	The RANSAC algorithm start with fixed number of iterations (2500) and  high inlier to total points ratio of 0.8. The Euclidian          distance threshold is set to 5 pixels between transformed coordinates and original coordinates to count a point pair as an inlier.      4 points are used to find the hypothesis at a time.
    •	Further, it checks for best hypothesis with inlier ratio over 0.8, if no hypothesis is found then the inlier ratio is reduced by        0.05 and the hypothesis search is recursive until the best hypothesis is returned
    •	Using the best hypothesis returned from RANSAC i.e. the transformation matrix the images are then stitched together to create a         panorama
    •	For creating the panorama, a blank canvas of the size of both images combined horizontally is created and then the first image is       stored to the left
    •	The second image is then filled into the canvas using inverse warping and bilinear interpolation using the inverse of the               transformation matrix
    •	For overlapping locations, the pixel values are averaged


