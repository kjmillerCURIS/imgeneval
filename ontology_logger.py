import os
import sys
import pprint


class Logger:

    def __init__(self):
        self.buffer = []
        self.pp = pprint.PrettyPrinter()

    def log(self, s, pretty=False):
        if pretty:
            s = self.pp.pformat(s)

        self.buffer.append(str(s))
        print(s)

    def clear(self):
        self.buffer = []

    def save_and_clear(self, filename, clear=True):
        f = open(filename, 'w')
        f.write('\n'.join(self.buffer) + '\n')
        f.close()
        if clear:
            self.clear()
