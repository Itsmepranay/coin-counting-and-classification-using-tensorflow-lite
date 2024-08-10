import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util

def count_first_words(folder_path):
    # Initialize a dictionary to keep track of word counts
    word_counts = {}

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file
    for file_name in files:
        # Make sure it's a text file
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                # Read all lines in the file
                lines = file.readlines()

                # Iterate through each line
                for line_number, line in enumerate(lines, 1):
                    # Split the line into words
                    words = line.split()
                    if words:
                        # Get the first word
                        first_word = words[0]

                        # Update the count for this word in the dictionary
                        if first_word in word_counts:
                            word_counts[first_word] += 1
                        else:
                            word_counts[first_word] = 1

                        # Print the first word of the line
                        print(f"File: {file_name}, Line {line_number}, First Word: {first_word}")

    # After processing all files, print the word counts
    print("\ncoin Counts:")
    total = 0
    for word, count in word_counts.items():
        print(f"{word}: {count}")
        if word == "one" or word == "one_0" or word == "one_1" or word == "one_2":
            denomination = 1
        elif word == "two" or word =="two_0" or word =="two_1" or word =="two_2" :
            denomination = 2
        elif word == "five" or word =="five_0" or word =="five_1" or word =="five_2" :
            denomination = 5
        elif word == "ten" or word == "ten_0" or word == "ten_1" or word == "ten_2" :
            denomination = 10
        elif word == "twenty" or word == "twenty_0":
            denomination = 20
        total = total + (denomination * count)
    print(f"Total: Rs {total}")


if __name__ == "__main__":
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in', required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
    parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir', default=None)
    parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.', default=None)
    parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder', action='store_true')
    parser.add_argument('--noshow_results', help='Don\'t show result images (only use this if --save_results is enabled)', action='store_false')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')
    parser.add_argument('--folder', help='Path to folder to count first words in text files', default=None)

    args = parser.parse_args()

    # Your existing code...
    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    use_TPU = args.edgetpu
    save_results = args.save_results
    show_results = args.noshow_results
    IM_NAME = args.image
    IM_DIR = args.imagedir

    # If both an image AND a folder are specified, throw an error
    if (IM_NAME and IM_DIR):
        print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
        sys.exit()

    # If neither an image or a folder are specified, default to using 'test1.jpg' for image name
    if (not IM_NAME and not IM_DIR):
        IM_NAME = 'test1.jpg'

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_TPU:
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'

    CWD_PATH = os.getcwd()

    if IM_DIR:
        PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
        images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.heic')
        if save_results:
            RESULTS_DIR = IM_DIR + '_results'

    elif IM_NAME:
        PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_NAME)
        images = glob.glob(PATH_TO_IMAGES)
        if save_results:
            RESULTS_DIR = 'results'

    if save_results:
        RESULTS_PATH = os.path.join(CWD_PATH, RESULTS_DIR)
        # Remove existing results folder if it exists
        if os.path.exists(RESULTS_PATH):
            import shutil
            shutil.rmtree(RESULTS_PATH)
        os.makedirs(RESULTS_PATH)

    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del(labels[0])

    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']

    if 'StatefulPartitionedCall' in outname:
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    for image_path in images:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        detections = []

        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(image, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

        if show_results:
            cv2.imshow('Object detector', image)

            if cv2.waitKey(0) == ord('q'):
                break

        if save_results:
            image_fn = os.path.basename(image_path)
            image_savepath = os.path.join(CWD_PATH, RESULTS_DIR, image_fn)
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn + '.txt'
            txt_savepath = os.path.join(CWD_PATH, RESULTS_DIR, txt_result_fn)

            cv2.imwrite(image_savepath, image)

            with open(txt_savepath, 'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

    if args.folder:
        folder_path = args.folder
        count_first_words(folder_path)

    cv2.destroyAllWindows()
