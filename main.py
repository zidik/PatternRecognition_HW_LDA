__author__ = 'Mark Laane'

import cv2
import numpy
import random
import logging
from loading_images import load_face_vectors_from_disk, detect_one_face
from sklearn.lda import LDA


no_of_persons = 13  # Number of persons
samples_person = 10  # Number of samples per person
samples_training = 9
image_size = (200, 200)  # All face images will be resized to this


def main():

    logging.basicConfig(format='[%(asctime)s] %(levelname)7s: %(message)s', level=logging.DEBUG)

    all_image_numbers = generate_all_image_numbers(no_of_persons, samples_person)
    classes = all_image_numbers[:, 0]
    all_face_vectors = load_face_vectors_from_disk(all_image_numbers, image_size)

    classifier = LDA()
    logging.debug("Training..")
    classifier.fit(all_face_vectors, classes)

    while True:
        function = input(
            "0)Exit\n"
            "1)Live test\n"
            "2)Test image \"test.JPG\"\n"
            "3)General test\n"
            "\n"
            "Choose function:"
        )
        if function == "1":
            test_live(classifier, all_face_vectors)
        elif function == "2":
            test_one_image(classifier, all_face_vectors)
        elif function == "3":
            test(all_face_vectors, classes)
        elif function == "0":
            return


def test_live(classifier, all_face_vectors):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            continue  # Let's just try again
        cv2.imshow('video', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            vector = extract_vectorized_face(frame)
        except UserWarning:
            logging.debug("No face was found..")
        else:
            prediction, probabilities = predict(classifier, vector)
            display_match(all_face_vectors, prediction-1, False)
            print(probabilities)

        key_no = cv2.waitKey(30) & 0xFF
        if key_no == ord('q'):
            logging.debug("'q' received - Quitting...")
            break
    cap.release()

def test_one_image(classifier, all_face_vectors):
    path = "./test.JPG"
    print("Loading \"{}\" ...".format(path))
    image = cv2.imread(filename=path, flags=0)
    try:
        vector = extract_vectorized_face(image)
    except UserWarning:
        logging.debug("No face was found..")
    else:
        prediction, probabilities = predict(classifier, vector)
        print("Prediction is: {}".format(prediction))
        display_match(all_face_vectors, prediction-1)
        print(probabilities)
    cv2.waitKey(20)

def test(all_face_vectors, classes):
    training_indices, testing_indices = select_indices_for_training_and_test()
    training_classes = classes[training_indices]
    training_data = all_face_vectors[training_indices]
    test_classes = classes[testing_indices]
    test_data = all_face_vectors[testing_indices]

    classifier = LDA()
    logging.debug("Training..")
    classifier.fit(training_data, training_classes)
    predictions, probabilities = predict(classifier, test_data)
    for prediction, probability in zip(predictions, probabilities):
        print("prediciton:{}, probability:{}".format(prediction, probability[prediction-1]))

    right_classification = predictions == test_classes
    prediction_rate = numpy.count_nonzero(right_classification) / right_classification.size
    logging.info("Prediction rate is: {:.2%}".format(prediction_rate))


def predict(classifier, test_data):
    treshold = 0.996
    logging.debug("Predicting...")
    predictions = classifier.predict(test_data)
    probabilities = classifier.predict_proba(test_data)

    #Mark all predictions below treshold with -1
    below_treshold = probabilities.max(axis=1) < treshold
    predictions[below_treshold] = -1
    return predictions, probabilities


def extract_vectorized_face(frame, show=True):
    try:
        face_image = detect_one_face(frame)
    except UserWarning:
        if show:
            message_frame = numpy.zeros(image_size)
            cv2.putText(message_frame, "No Face Found", (0, image_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
            cv2.imshow('face image', message_frame)
            cv2.waitKey(1)
        raise
    else:
        resized = cv2.resize(face_image, image_size)
        if show:
            cv2.imshow('face image', resized)
            cv2.waitKey(1)
        vector = numpy.ravel(resized).astype(numpy.float32, copy=False) / 255
    return vector


def display_match(all_face_vectors, person_index, wait_for_key=True):
    #Person index starts from 0

    if person_index < 0:
        matching_images = numpy.zeros(image_size)
        cv2.putText(matching_images,"Below Treshold", (0, image_size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255)
    else:
        images_in_row = 4
        row_list = []
        for start_no in range(0, samples_person, images_in_row):

            face_list = []
            for sample_no in range(start_no, start_no+images_in_row):
                if sample_no < samples_person:
                    face_vector = all_face_vectors[person_index*samples_person+sample_no]
                else:
                    face_vector = numpy.zeros_like(all_face_vectors[0])
                face_image = face_vector.copy().reshape(image_size)
                face_list.append(face_image)
            row_of_faces = numpy.hstack(face_list)
            row_list.append(row_of_faces)

        matching_images = numpy.vstack(row_list)

    cv2.imshow('MATCH', matching_images)

    if wait_for_key:
        while cv2.waitKey(20) == -1:
           pass


def select_indices_for_training_and_test():
    training_indices = []
    testing_indices = []
    for person_no in range(no_of_persons):
        random_permutation = random.sample(range(samples_person), samples_person)
        training_indices.extend(element + person_no*samples_person for element in random_permutation[:samples_training])
        testing_indices.extend(element + person_no*samples_person for element in random_permutation[samples_training:])
    return training_indices, testing_indices


def generate_all_image_numbers(no_of_persons, samples_person):
    """
    Generates and returns a list of all possible combinations of imagenumbers

    :param no_of_persons: number of persons
    :param samples_person: number of samples used per person
    :return: array of numbers
    """
    return numpy.mgrid[1:samples_person+1, 1:no_of_persons+1].T.reshape(-1, 2)[:, ::-1]


if __name__ == '__main__':
    main()