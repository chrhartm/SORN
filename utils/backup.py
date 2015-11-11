from __future__ import division
from pylab import *
#imports for file handling:
#~ import random
import os
from os.path import *
import shutil
import datetime
import time
try:
    from mpi4py import MPI
    imported_mpi = True
except ImportError:
    imported_mpi = False
import sys


######################################################################
#                         file handling stuff                        #
######################################################################
def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

current_time_int = int(time.mktime(time.gmtime()))
current_time = _now()
curr_string = ".."
dest_string = "../backup"

curr_directory = None
dest_directory = None
initialized = False

copied_files = []

def initialise_backup(mount=None,dest=None):
    """When backing things up from a folder different from the default use this call.
    Example use (for calling from a subfolder):
    import utils
    utils.initialise_backup(mount="../../", dest="../../backup")
    utils.backup()
    This does not need to be called when using the default locations
    """
    global curr_string, dest_string
    curr_string = mount
    dest_string = dest

def _initialize(filename):
    #initialize should only get called once; by the first call to utils.start
    global dest_directory, curr_directory, initialized
    #Calc dest_directory
    abs_filename = abspath(filename)
    (_,final_name) = split(abs_filename)
    (scriptname,_) = splitext(final_name)

    dest_base = abspath(expanduser(dest_string))
    dest_directory = join(dest_base,scriptname,current_time)
    if (not imported_mpi or (MPI.COMM_WORLD.rank == 0)) and (not exists(dest_directory)):
        try:
            os.makedirs(dest_directory)
        except OSError as e:
            print e
    #calc curr_directory
    curr_directory = abspath(expanduser(curr_string))
    initialized = True

def track_files(fn):
    def wrapped(*args, **kwargs):
        filename = args[0]
        abs_file = abspath(filename)
        if abs_file in copied_files:
            return
        fn(*args,**kwargs)
        copied_files.append(abs_file)
    return wrapped

def check_should_backup(fn):
    def wrapped(*args, **kwargs):
        if (imported_mpi and not MPI.COMM_WORLD.rank == 0):
            return
        fn(*args,**kwargs)
    return wrapped

@check_should_backup
@track_files
def copy(filename):
    final_filename = logfilename(filename)
    #Now test if destination directory exists:
    (final_dir,final_name) = split(final_filename)
    if not exists(final_dir):
        try:
            os.makedirs(final_dir)
        except OSError as e:
            print e
    #Now copy!
    try:
        shutil.copy2(filename,final_filename)
    except OSError as e:
        print e

def copy_source(filename):
    if filename[-1]=='c':  #Cheap hack to exclude saving pyc files
        filename = filename[:-1] #Assume source is always present
    copy(filename)

@check_should_backup
@track_files
def copy_directory(src):

    final_dir = logfilename(src)
    try:
        shutil.copytree(src,final_dir,symlinks=True)
    except OSError as e:
        print e

def backup(filename, start_seed=None):
    """The first time backup() is called, it notes the time and creates a
    suitable directory for all further files to be stored.  It copies
    utils.py into the directory, and also initialises the random seed,
    while also making a copy of the random seed.

    Successive calls to backup() result in only the source code being
    copied."""

    if not initialized:
        _initialize(filename)
        copy_directory(join(curr_string,'utils')) #<-- Don't forget to copy utils directory
        #~ copy_source(__file__) #<-- Don't forget to copy utils.py!
        if imported_mpi:
            rank = MPI.COMM_WORLD.rank
        else:
            rank = 0
        if start_seed == None:
            start_seed = current_time_int*(rank + 1)

        seed(start_seed)
        if rank == 0:
            try:
                f = file(logfilename("seed"),'w')
                f.write(str(start_seed))
                f.close()
            except OSError as e:
                print e
    copy_source(filename)

def logfilename(filename):
    if not initialized:
        print "Error - Script has not been backed up!"
        return filename
    #Calc file location and destination
    abs_filename = abspath(filename)
    if not abs_filename.startswith(curr_directory):
        print "Error - trying to backup file above main directory?!?"
        return filename
    reduced_filename = abs_filename[len(curr_directory)+1:]
    #Make the directory if necessary
    final_filename = join(dest_directory, reduced_filename)
    (final_dir,final_name) = split(final_filename)
    if not exists(final_dir):
        try:
            os.makedirs(final_dir)
        except OSError as e:
            print e
    return final_filename

def saveplot(figurename, f = None):
    '''This is intended to standardise the formatting and
        file save behaviour for creating plots '''
    if f == None:
        f = gcf()
    savefig(logfilename(figurename), dpi=300)
