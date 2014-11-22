__author__ = 'Mark'

import numpy
import cv2
import os
import errno


def load_face_vectors_from_disk(image_numbers, img_size, show=True):
    """
    Loads images from disk, detects faces from them, resizes the face images to common size, vectorizes the face image
    and stores it in a dictionary with key (pers_no, sample_no)

    :param image_numbers: List of tuples of image numbers to load. Tuples are in format: (pers_no, sample_no)
    :param img_size: Tuple (x, y). The detected faces are resized to this size.
    :param show: Boolean switch. Show detected and resized face image
    :return: Dictionary of face vectors.
    """
    number_of_images = len(image_numbers)
    pixels_in_image = img_size[0] * img_size[1]
    face_vectors = numpy.empty((number_of_images, pixels_in_image))

    for count, (person_no, sample_no) in enumerate(image_numbers):
        directory = "./ImageDatabase/"
        cache_directory = "./FaceCache/"
        filename = "f_{}_{:02}.JPG".format(person_no, sample_no)

        make_sure_path_exists(cache_directory)

        cached_path = "{}{}".format(cache_directory, filename)
        print("Loading cached face \"{}\" ...".format(cached_path))
        face_image = cv2.imread(filename=cached_path, flags=0)
        if face_image is None:
            print("Could not load cached face")
            path = "{}{}".format(directory, filename)
            print("Loading original \"{}\" ...".format(path))
            image = cv2.imread(filename=path, flags=0)
            if image is None:
                raise UserWarning("\"{}\" was not found ...".format(path))

            face_image = detect_one_face(image)
            cv2.imwrite(cached_path, face_image)

        resized = cv2.resize(face_image, img_size)

        if show:
            cv2.imshow('face image', resized)
            cv2.waitKey(1)

        vector = numpy.ravel(resized)
        face_vectors[count] = vector.astype(face_vectors.dtype, copy=False) / 255
    return face_vectors


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


def detect_one_face(image):
    """
    Detects one image from input and returns the face
    :param image: Input image
    :return: Face image
    """
    faces = faceCascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(image.shape[1] // 10, image.shape[0] // 10)
    )
    number_of_faces = len(faces)
    if number_of_faces == 0:
        raise UserWarning("No face was found on image")
    elif number_of_faces == 1:
        #One face was found
        x, y, w, h = faces[0]
    else:
        raise UserWarning("Multiple faces were found on image")

    face_image = image[y:y + h, x:x + w]
    return face_image


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise