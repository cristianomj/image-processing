import cv2
import getopt
import numpy as np
import urllib
import sys

MIN_MATCH_COUNT = 10


def feature_match(source, template):
    # Initialize SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the key-points and descriptors with SIFT
    source_key_points, source_descriptors = sift.detectAndCompute(source, None)
    template_key_points, template_descriptors = sift.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(template_descriptors, source_descriptors, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        source_pts = np.float32([template_key_points[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        destination_pts = np.float32([source_key_points[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(source_pts, destination_pts, cv2.RANSAC, 5.0)

        theta = - np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        rotation = 0

        if (theta > -55) & (theta < -35):
            rotation = 45
        elif (theta > -100) & (theta < -80):
            rotation = 90
        elif (theta > -145) & (theta < -125):
            rotation = 135
        elif (theta > 170) & (theta < 190):
            rotation = 180
        elif (theta > 125) & (theta < 145):
            rotation = 225
        elif (theta > 80) & (theta < 100):
            rotation = 270
        elif (theta > 35) & (theta < 55):
            rotation = 315

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # get coordinates rounded to nearest integer (these will still be floats)
        coordinates = np.rint(cv2.perspectiveTransform(pts, M))

        # convert these to ints
        x_coordinates = []
        y_coordinates = []
        for i in range(len(coordinates)):
            x_coordinates.append(0 if int(coordinates[i][0][0]) < 0 else int(coordinates[i][0][0]))
            y_coordinates.append(0 if int(coordinates[i][0][1]) < 0 else int(coordinates[i][0][1]))

        crop_left = min(x_coordinates)
        crop_top = min(y_coordinates)
        crop_right = max(x_coordinates)
        crop_bottom = max(y_coordinates)

        print (crop_left, crop_top, crop_right, crop_bottom, rotation, len(good_matches))
        sys.exit()

    else:
        print >> sys.stderr, "Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)
        sys.exit(2)


def fetch_and_decode_image(url):
    req = urllib.urlopen(url)

    if req.getcode() != 200:
        print >> sys.stderr, "Image not found - %s" % url
        sys.exit(2)

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, 0)

    return img


def main(argv):
    source_url = ''
    template_url = ''

    try:
        opts, args = getopt.getopt(argv, "s:t:", ["source=", "template="])
    except getopt.GetoptError:
        print >> sys.stderr, 'featureMatching.py -s <source_url> -t <template_url>'
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-s", "--source"):
            source_url = arg
        elif opt in ("-t", "--template"):
            template_url = arg

    source = fetch_and_decode_image(source_url)
    template = fetch_and_decode_image(template_url)
    feature_match(source, template)

if __name__ == '__main__':
    main(sys.argv[1:])
