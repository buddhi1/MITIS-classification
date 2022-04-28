import randomforest
import naivebayes
import svm
import nn_train
import cnn
import sys


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print('Models not requested. Select ''all'' or atleast one from model list.')
        sys.exit()
    elif len(sys.argv) == 2:
        if sys.argv[1] == 'all':
            randomforest.run_loop()
            naivebayes.run_loop()
            svm.run_loop()
            nn_train.run_loop()
            cnn.run(True)
    if 'randomforest' in sys.argv:
        randomforest.run_loop()
    if 'naivebayes' in sys.argv:
        naivebayes.run_loop()
    if 'svm' in sys.argv:
        svm.run_loop()
    if 'nn' in sys.argv:
        nn_train.run_loop()
    if 'cnn' in sys.argv:
        if 'load' in sys.argv:
            cnn.run(False)
        else:
            cnn.run(True)
            