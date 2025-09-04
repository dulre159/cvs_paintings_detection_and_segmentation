import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import cv2
import timeit
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def getFeaturesVectorResNet18ForBanchmarkGPU(img,  resnet18Model):
    img = img.cuda()
    features_var = resnet18Model(img).cpu()# get the output from the last hidden layer of the pretrained resnet
    features = features_var.data  # get the tensor out of the variable
    return features

def getFeaturesVectorResNet50ForBanchmarkGPU(img,  resnet50Model):
    img = img.cuda()
    features_var = resnet50Model(img).cpu()# get the output from the last hidden layer of the pretrained resnet
    features = features_var.data  # get the tensor out of the variable
    return features

def banchMarkOrbUsingBFMatcherWithKnn(paintingsDB, orb, listOfGrayInputImages, bruteForcematcher,
                                      match_ratio=0.75, k=2):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = bruteForcematcher.knnMatch(inputImageDescriptor, tmpImageDescriptor, k)
            good = []
            # The matches with shorter distance are the ones we want.
            # match_ratio = 0.8  # Nearest neighbor matching ratio
            for m, n in matches:
                if m.distance < match_ratio * n.distance:
                    good.append([m])
            scoresDictionary[imagefile] = len(good)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict

def banchMarkOrbUsingBFMatcher(paintingsDB, orb, listOfGrayInputImages, bruteForcematcher):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = bruteForcematcher.match(inputImageDescriptor, tmpImageDescriptor)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            scoresDictionary[imagefile] = len(matches)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict


def banchMarkOrbUsingFLANNMatcherWithKnn(paintingsDB, orb, listOfGrayInputImages, flann, match_ratio=0.75, k=2):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = flann.knnMatch(inputImageDescriptor, tmpImageDescriptor, k)
            good = []
            # The matches with shorter distance are the ones we want.
            # match_ratio = 0.8  # Nearest neighbor matching ratio
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                (m, n) = m_n
            # for m, n in matches:
                if m.distance < match_ratio * n.distance:
                    good.append([m])
            scoresDictionary[imagefile] = len(good)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict

def banchMarkOrbUsingFLANNMatcher(paintingsDB, orb, listOfGrayInputImages, flann):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = flann.match(inputImageDescriptor, tmpImageDescriptor)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            scoresDictionary[imagefile] = len(matches)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict


def banchMarkAKAZEUsingBFMatcherWithKnn(paintingsDB, akaze, listOfGrayInputImages, bruteForcematcher,
                                        match_ratio=0.75, k=2):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = bruteForcematcher.knnMatch(inputImageDescriptor, tmpImageDescriptor, k)
            good = []
            # The matches with shorter distance are the ones we want.
            # match_ratio = 0.8  # Nearest neighbor matching ratio
            for m, n in matches:
                if m.distance < match_ratio * n.distance:
                    good.append([m])
            scoresDictionary[imagefile] = len(good)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict


def banchMarkAKAZEUsingBFMatcher(paintingsDB, akaze, listOfGrayInputImages, bruteForcematcher):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = bruteForcematcher.match(inputImageDescriptor, tmpImageDescriptor)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            scoresDictionary[imagefile] = len(matches)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict


def banchMarkAKAZEUsingFLANNMatcherWithKnn(paintingsDB, akaze, listOfGrayInputImages, flann, match_ratio=0.75,
                                           k=2):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = flann.knnMatch(inputImageDescriptor, tmpImageDescriptor, k)
            good = []
            # The matches with shorter distance are the ones we want.
            # match_ratio = 0.8  # Nearest neighbor matching ratio
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                (m, n) = m_n
            # for m, n in matches:
                if m.distance < match_ratio * n.distance:
                    good.append([m])
            scoresDictionary[imagefile] = len(good)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict

def banchMarkAKAZEUsingFLANNMatcher(paintingsDB, akaze, listOfGrayInputImages, flann):
    resultDict = {}
    for inputimagefile, inputImage in listOfGrayInputImages.items():
        # Get input image descripots and keypoints
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImage[2], None)
        scoresDictionary = {}
        for imagefile, painting in paintingsDB.items():
            tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(painting[2], None)
            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = flann.match(inputImageDescriptor, tmpImageDescriptor)
            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            scoresDictionary[imagefile] = len(matches)
        resultDict[inputimagefile] = scoresDictionary
    return resultDict

def banchmarkUsingResNet18GPU(paintingsDBWithPILImages, listOfPILInputImages, resnet18Model):
    resultDict = {}
    for inputimagefile, inputImage in listOfPILInputImages.items():
        # Get input image descripots and keypoints
        inputFeeatureVector = getFeaturesVectorResNet18ForBanchmarkGPU(inputImage[3],resnet18Model)
        scoresDictionary = {}
        for imagefile, painting in paintingsDBWithPILImages.items():
            tmpImageFeeatureVector = getFeaturesVectorResNet18ForBanchmarkGPU(painting[3], resnet18Model)
            scoresDictionary[imagefile] = \
                cosine_similarity(inputFeeatureVector.reshape((1, -1)), tmpImageFeeatureVector.reshape((1, -1)))[0][0]
        resultDict[inputimagefile] = scoresDictionary
    return resultDict


def banchmarkUsingResNet50GPU(paintingsDBWithPILImages, listOfPILInputImages, resnet50Model):
    resultDict = {}
    for inputimagefile, inputImage in listOfPILInputImages.items():
        # Get input image descripots and keypoints
        inputFeeatureVector = getFeaturesVectorResNet50ForBanchmarkGPU(inputImage[3],resnet50Model)
        scoresDictionary = {}
        for imagefile, painting in paintingsDBWithPILImages.items():
            tmpImageFeeatureVector = getFeaturesVectorResNet50ForBanchmarkGPU(painting[3], resnet50Model)
            scoresDictionary[imagefile] = \
                cosine_similarity(inputFeeatureVector.reshape((1, -1)), tmpImageFeeatureVector.reshape((1, -1)))[0][0]
        resultDict[inputimagefile] = scoresDictionary
    return resultDict

def measureAccuracy(resultDict):
    tot = 0
    totFound = 0
    for inputImageFile, scoresDictionary in resultDict.items():
        tot += 1
        originalPaintingFileName = inputImageFile.split("_")[1].split(".")[0]
        sortedScoresDictionary = {k: v for k, v in sorted(scoresDictionary.items(), key=lambda item: item[1], reverse=True)}
        retrievedImageFilename = list(sortedScoresDictionary.keys())[0].split(".")[0]
        if originalPaintingFileName == retrievedImageFilename:
            totFound +=1
    return totFound/tot


def wrapper(func, *args, **kwargs):
    output_container = []
    def wrapped():
        output_container.append(func(*args, **kwargs))
    timer = timeit.Timer(wrapped)
    delta = timer.timeit(1)
    return delta, output_container.pop()

def doBanchmarking(DBPath, inputImagesPath):
    inputimagesDic = {}

    # 2. Create a PyTorch Variable with the transformed image
    preprocess = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor()#,
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for imagefile in os.listdir(inputImagesPath):
        tmpImg = cv2.imread(os.path.join(inputImagesPath, imagefile))
        tmpImgRGB = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
        tmpImgGray = cv2.cvtColor(tmpImgRGB, cv2.COLOR_RGB2GRAY)
        tmpImgPIL0 = Image.fromarray(tmpImgRGB)
        #tmpImgPIL = Variable(preprocess(tmpImgPIL0).unsqueeze(0))
        tmpImgPIL = Variable(torch.unsqueeze(preprocess(tmpImgPIL0), dim=0).float(), requires_grad=False)
        inputimagesDic[imagefile] = (tmpImg, tmpImgRGB, tmpImgGray, tmpImgPIL)

    paintingsDB = {}

    for imagefile in os.listdir(DBPath):
        tmpImg = cv2.imread(os.path.join(DBPath, imagefile))
        tmpImgRGB = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
        tmpImgGray = cv2.cvtColor(tmpImgRGB, cv2.COLOR_RGB2GRAY)
        tmpImgPIL0 = Image.fromarray(tmpImgRGB)
        #tmpImgPIL = Variable(preprocess(tmpImgPIL0).unsqueeze(0))
        tmpImgPIL = Variable(torch.unsqueeze(preprocess(tmpImgPIL0), dim=0).float(), requires_grad=False)
        paintingsDB[imagefile] = (tmpImg, tmpImgRGB, tmpImgGray, tmpImgPIL)

    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    bfwcc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Create FLANN based matcher with the parameters suggested for ORB
    FLANN_INDEX_LSH = 6
    #FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12 #6
                        key_size=12,  # 20 #12
                        multi_probe_level=1)  # 2 #1
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 2 #1
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    resnet18Model = models.resnet18(pretrained=True)
    modules = list(resnet18Model.children())[:-1]
    resnet18Model = nn.Sequential(*modules)
    for p in resnet18Model.parameters():
        p.requires_grad = False
    resnet18Model = resnet18Model.cuda()



    resnet50Model = models.resnet50(pretrained=True)
    modules = list(resnet50Model.children())[:-1]
    resnet50Model = nn.Sequential(*modules)
    for p in resnet50Model.parameters():
        p.requires_grad = False
    resnet50Model = resnet50Model.cuda()

    k = 2
    match_ratio = 0.75


    # brn18_delta, brn18_ret = wrapper(banchmarkUsingResNet18GPU, paintingsDB, inputimagesDic, resnet18Model)
    # print("ResNet18 - Time: %f - Accuracy: %f" %(brn18_delta,measureAccuracy(brn18_ret)))
    #
    # brn50_delta, brn50_ret = wrapper(banchmarkUsingResNet50GPU, paintingsDB, inputimagesDic, resnet50Model)
    # print("ResNet50 - Time: %f - Accuracy: %f" % (brn50_delta, measureAccuracy(brn50_ret)))

    # orbbfknn_delta, orbbfknn_ret = wrapper(banchMarkOrbUsingBFMatcherWithKnn, paintingsDB, orb, inputimagesDic, bf, match_ratio, k)
    # print("ORBBFKNN - Time: %f - Accuracy: %f" % (orbbfknn_delta, measureAccuracy(orbbfknn_ret)))

    banchMarkOrbUsingBFMatcherWithKnn(paintingsDB, orb, inputimagesDic, bf, match_ratio, k)

    # orbbf_delta, orbbf_ret = wrapper(banchMarkOrbUsingBFMatcher, paintingsDB, orb, inputimagesDic, bfwcc)
    # print("ORBBF - Time: %f - Accuracy: %f" % (orbbf_delta, measureAccuracy(orbbf_ret)))
    #
    # orbflannknn_delta, orbflannknn_ret = wrapper(banchMarkOrbUsingFLANNMatcherWithKnn, paintingsDB, orb, inputimagesDic, flann, match_ratio, k)
    # print("ORBFLANNKNN - Time: %f - Accuracy: %f" % (orbflannknn_delta, measureAccuracy(orbflannknn_ret)))
    #
    # orbflann_delta, orbflann_ret = wrapper(banchMarkOrbUsingFLANNMatcher, paintingsDB, orb, inputimagesDic, flann)
    # print("ORBFLANN - Time: %f - Accuracy: %f" % (orbflann_delta, measureAccuracy(orbflann_ret)))

    # akazebfknn_delta, akazebfknn_ret = wrapper(banchMarkAKAZEUsingBFMatcherWithKnn, paintingsDB, akaze, inputimagesDic, bf, match_ratio, k)
    # print("AKAZEBFKNN - Time: %f - Accuracy: %f" % (akazebfknn_delta, measureAccuracy(akazebfknn_ret)))

    banchMarkAKAZEUsingBFMatcherWithKnn(paintingsDB, akaze, inputimagesDic, bf, match_ratio, k)
    #
    # akazebf_delta, akazebf_ret = wrapper(banchMarkAKAZEUsingBFMatcher, paintingsDB, akaze, inputimagesDic, bfwcc)
    # print("AKAZEBF - Time: %f - Accuracy: %f" % (akazebf_delta, measureAccuracy(akazebf_ret)))
    #
    # akazeflannknn_delta, akazeflannknn_ret = wrapper(banchMarkAKAZEUsingFLANNMatcherWithKnn, paintingsDB, akaze, inputimagesDic, flann, match_ratio, k)
    # print("AKAZEFLANNKNN - Time: %f - Accuracy: %f" % (akazeflannknn_delta, measureAccuracy(akazeflannknn_ret)))
    #
    # akazeflann_delta, akazeflann_ret = wrapper(banchMarkAKAZEUsingFLANNMatcher, paintingsDB, akaze, inputimagesDic, flann)
    # print("AKAZEFLANN - Time: %f - Accuracy: %f" % (akazeflann_delta, measureAccuracy(akazeflann_ret)))

    # print(brn18_delta)
    # print(orbbfknn_delta)
    # print(orbflannknn_delta)
    # print(akazebfknn_delta)
    # print(akazeflannknn_delta)
    # print(brn18_ret)

if __name__ == "__main__":
    doBanchmarking("paintings_db", "retrieval_benchmark_images")