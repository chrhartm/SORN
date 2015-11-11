from __future__ import division
from pylab import *
#imports for file handling:
import random
import os
from os.path import *
import shutil
import datetime
import time
try:
    from mpi4py import MPI
except ImportError:
    pass
#imports for param_file
#~ import argparse
import sys
from optparse import OptionParser

######################################################################
#                 Param file stuff                                   #
######################################################################

sys.path.insert(0, './params')

try:
    with open("help.txt", 'r') as help_file:
        help_text = help_file.read()
except EnvironmentError:
    help_text = "Scientific research script."
usage = "%prog [options] params"
#~ #argparse code for python 3.3+
#~ parser = argparse.ArgumentParser(description=help_text)
#~ parser.add_argument('f', metavar='FILE', type=string,
                   #~ help='    parameter file')
#~ def param_file():
    #~ args = parser.parse_args()
    #~ return args.param_file

parser = OptionParser(description=help_text, usage=usage)
#~ parser.add_option("-p", "--param", action="store", type="string",
                #~ dest="param", help="name of param module", metavar="MODULE")

def param_file():
    (options, args) = parser.parse_args()
    #~ if options.param is None:
    if len(args) < 1:
        parser.error("param file not set!")
    return args[0]
