import os
import subprocess
import tarfile

import wget


class Meteor:
    """ Implementation of a wrapper for the Meteor metric provided by
        http://www.cs.cmu.edu/~alavie/METEOR/
        Instructions about the command http://www.cs.cmu.edu/~alavie/METEOR/README.html

        References:
            - https://github.com/tsenghungchen/SA-tensorflow
            - https://github.com/EdinburghNLP/nematus/blob/a956559879003e569f53d82b73fc489b87e1fa87/nematus/metrics/meteor.py
            - https://github.com/gcunhase/NLPMetrics
    """

    def __init__(self, file_hypothesis, file_reference, meteor_path,
                 meteor_jar_file="meteor-1.5.jar", tokenize_words=False, verbose=True):
        if not os.path.isdir(meteor_path):
            self.download(destination_folder=meteor_path)

        self.meteor_cmd = ['java', '-Xmx2G', '-jar', meteor_path + meteor_jar_file,
                           file_hypothesis, file_reference, '-l', 'en']
        print(f"Meteor command: <{self.meteor_cmd}>")
        if tokenize_words:
            self.meteor_cmd.append('-norm')

        self.meteor_process = subprocess.Popen(self.meteor_cmd,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output = self.meteor_process.stdout.read().decode('ascii')
        if verbose:
            print(output)
        self.score = float(output.split("Final score:")[-1])
        print("Meteor: {:.2f}".format(self.score))

        # TODO calculate std

        # Used to guarantee thread safety
        # self.lock = threading.Lock()

    def download(self, destination_folder):
        meteor_jar_url = 'http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz'

        wget.download(meteor_jar_url)
        with tarfile.open("meteor-1.5.tar.gz", "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(tar, path=destination_folder)
