# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2 as cv
def main():
    print('Morphological')
    name_file = 'C:/Users/dilomere/PycharmProjects/CardDetection/data/image5.jpg'
    #name_file = 'image5.jpg'

    # opening an image
    img = cv.imread(name_file, cv.IMREAD_GRAYSCALE)
    (M, N) = (img.shape[0], img.shape[1])
    cv.imshow('image - original', img)
    cv.waitKey(0)


    # remove noise
    sigma = 0.8
    img_gauss = cv.GaussianBlur(img, (5, 5), sigmaX=sigma)
    #cv.imshow('image - gaussian blur', img_gauss)
    #cv.waitKey(0)

    # canny edge detection
    canny = cv.Canny(img_gauss, 100, 150, L2gradient=True)
    #cv.imshow('canny', canny)
    #cv.waitKey(0)


    # morphological operations
    struct_elem = np.ones((5, 5), np.uint8)
    dilated = cv.dilate(canny, struct_elem, iterations=1)
    eroded = cv.erode(dilated, struct_elem, iterations=1)

    #cv.imshow('dilated', dilated)
    #cv.imshow('eroded', eroded)

    # find outer contours
    img_color = cv.imread(name_file, cv.IMREAD_COLOR)
    img_contours = img_color.copy()

    contours, hierarchy = cv.findContours(eroded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cv.drawContours(img_color, contours, i, (128 - i * 16, 127 + i * 16, 255), thickness=2)

    # contours, hierarchy = cv.findContours(eroded, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    # for i in range(0, len(contours)):
    #     cv.drawContours(img_color, contours, i, (128 - i*4,  127 + i*4, 255), thickness=0)
    count = 0
    #cv.imshow('contours', img_color)
    for poly in contours:
        cv.fillPoly(img_color, np.array([poly], dtype=np.int32), (255, 0, 0))

    resultimagescale = cv.resize(img_color, None, fx=0.3, fy=0.3)
    cv.imshow('rescale', resultimagescale)
    cv.waitKey(0)

    #cv.imshow("results", img_color)
    #cv.waitKey(0)

    print('# of outer contours found: ', len(contours))
    for c in contours:
        if cv.contourArea(c) > 500:
            count += 1

    cv.imwrite('gray.jpg', img_color)

    img_gray = cv.imread('gray.jpg', cv.IMREAD_GRAYSCALE)

    print('There are {} cards'.format(count))
    #cv.waitKey(0)
    corners = cv.goodFeaturesToTrack(img_gray, 4, 0.5, 50)
    print(corners)


    for corner in corners:
        x, y = corner.ravel()
        cv.circle(img_color, (int(x), int(y)), 5, (36, 255, 12), -1)



    #cv.imshow('image', img_color)
    #cv.waitKey()
    destpts = np.float32([[400,0],[0,0],[0,200],[400,200]])
    # applying PerspectiveTransform() function to transform the perspective of the given source image to the corresponding points in the destination image
    resmatrix = cv.getPerspectiveTransform(corners, destpts)
    # applying warpPerspective() function to display the transformed image
    resultimage = cv.warpPerspective(img, resmatrix, (500, 600))
    # displaying the original image and the transformed image as the output on the screen
    #cv.imshow('frame', img)
    #cv.imshow('frame1', resultimage)
    #cv.waitKey(0)

    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/