# -*- coding: utf-8 -*-
"""

"""

from abc import ABCMeta, abstractmethod

from os.path import isfile
from time import strftime

import numpy as np
import sys
try:
    from mpi4py import MPI
    imported_mpi = True
except ImportError:
    imported_mpi = False
from autotable import AutoTable

#Quickly patch the code for mpi
def only_root(fn):
    def wrapped(*args, **kwargs):
        if not imported_mpi or MPI.COMM_WORLD.rank == 0:
            fn(*args, **kwargs)
    return wrapped

@only_root
def pprint(obj="", end='\n'):
    """
    Parallel print: Make sure only one of the MPI processes
    calling this function actually prints something. All others
    (comm.rank != 0) return without doing enything.
    """
    if isinstance(obj, str):
        sys.stdout.write(obj+end)
    else:
        sys.stdout.write(repr(obj)+end)
    sys.stdout.flush()

#=============================================================================
# DataHandler (AbstractBaseClass)

class DataHandler(object):
    __metaclass__ = ABCMeta

    """ Base class for handler which can be set to handle incoming data by DataLog."""
    def __init__(self):
        pass

    def register(self, tblname):
        """ Called by Datalog whenever this object is set as an handler for some table """
        pass

    @abstractmethod
    def append(self, tblname, value):
        pass

    def extend(self, valdict):
        for key, val in valdict.items():
            self.append(key, val)

    def remove(self, tblname):
        pass

    def close(self):
        pass

#=============================================================================
# StoreToH5 Handler

class StoreToH5(DataHandler):
    default_autotbl = None

    @only_root
    def __init__(self, destination=None):
        """
        Store data to the specified .h5 destination.

        *destination* may be either a file name or an existing AutoTable object
        """
        self.destination = destination

        if isinstance(destination, AutoTable):
            self.autotbl = destination
        elif isinstance(destination, str):
            self.autotbl = AutoTable(destination)
        elif destination is None:
            if StoreToH5.default_autotbl is None:
                self.autotbl = AutoTable()
            else:
                self.autotbl = StoreToH5.default_autotbl
        else:
            raise TypeError("Expects an AutoTable instance or a string as argument")

        if StoreToH5.default_autotbl is None:
            StoreToH5.default_autotbl = self.autotbl
    def __repr__(self):
        if "destination" in self.__dict__:
            return "StoreToH5 into file %s" % self.destination
        else:
            return "Uninitialised hf5 logger"
    @only_root
    def append(self, tblname, value):
        self.autotbl.append(tblname, value)
    @only_root
    def extend(self, valdict):
        self.autotbl.extend(valdict)
    @only_root
    def close(self):
        self.autotbl.close()


#=============================================================================
# StoreToTxt Handler

class StoreToTxt(DataHandler):

    @only_root
    def __init__(self, destination=None):
        """
        Store data to the specified .txt destination.

        *destination* has to be a file name
        """
        if isinstance(destination, str):
            self.txt_file = open(destination, 'w')
        elif destination is None:
            if not isfile('terminal.txt'):
                self.txt_file = open('terminal.txt', 'w')
            else:
                raise ValueError("Please enter a file name that does not already exist.")

    @only_root
    def append(self, tblname, value):
        self.txt_file.write("%s = %s\n" % (tblname, value))

    @only_root
    def close(self):
        self.txt_file.close()

#=============================================================================
# TextPrinter Handler

class TextPrinter(DataHandler):
    def __init__(self):
        pass

    def append(self, tblname, value):
        pprint("  %8s = %s " % (tblname, value))

#=============================================================================
# DataLog

class DataLog:
    def __init__(self):
        self.policy = []             # Ordered list of (tbname, handler)-tuples
        self._lookup_cache = {}      # Cache for tblname -> handlers lookups

    def _lookup(self, tblname):
        """ Return a list of handlers to be used for tblname """
        if tblname in self._lookup_cache:
            return self._lookup_cache[tblname]

        handlers = []
        for (a_tblname, a_handler) in self.policy:
            if a_tblname == tblname or a_tblname == "*": # XXX wildcard matching XXX
                handlers.append(a_handler)
        self._lookup_cache[tblname] = handlers
        return handlers

    @only_root
    def progress(self, message, completed=None):
        """ Append some progress message """
        if completed == None:
            print "[%s] %s" % (strftime("%H:%M:%S"), message)
        else:
            totlen = 65-len(message)
            barlen = int(totlen*completed)
            spacelen = totlen-barlen
            print "[%s] %s [%s%s]" % (strftime("%H:%M:%S"), message, "*"*barlen, "-"*spacelen)

    @only_root
    def append(self, tblname, value):
        """ Append the given value and call all the configured DataHandlers."""
        for h in self._lookup(tblname):
            h.append(tblname, value)

    @only_root
    def extend(self, valdict):
        """
        Append all entries in the dictionary and call all the configured DataHandlers

        *valdict* is expected to be a dictionary of key-value pairs.
        """
        # Construct a set with all handlers to be called
        all_handlers = set()
        for tblname, val in valdict.items():
            hl = self._lookup(tblname)
            all_handlers = all_handlers.union(hl)

        # Call all handlers but create a personalized version
        # of valdict with oble the values this particular handler
        # is interested in
        for handler in all_handlers:
            argdict = {}
            for tblname, val in valdict.items():
                hl = self._lookup(tblname)

                if handler in hl:
                    argdict[tblname] = val

            handler.extend(argdict)

    def ignored(self, tblname):
        """
        Returns True, then the given *name* is neither stored onto disk,
        nor visualized or triggered upon. When *ignored('something')* returns
        True, it will make no difference if you *append* a value to table *tblname* or not.

        This can be especially useful when running a (MPI-)parallel programs and collecting
        the value to be logged is an expensive operation.

        Example::

            if not dlog.ignored('summed_data'):
                summed_data =  np.empty_like(data)
                mpicomm.Reduce((data, MPI.DOUBLE), (summed_data, MPI_DOUBLE), MPI.SUM)
                dlog.append('summed_data', summed_data)

            [..]
        """
        return self._lookup(tblname) == []

    def set_handler(self, tblname, handler_class, *args, **kargs):
        """ Set the specifies handler for all data stored under the name *tblname* """

        if not issubclass(handler_class, DataHandler):
            raise TypeError("handler_class must be a subclass of DataHandler ")

        handler = handler_class(*args, **kargs)             # instantiate handler
        handler.register(tblname)

        if isinstance(tblname, str):
            self.policy.append( (tblname, handler) )    # append to policy
        elif hasattr(tblname, '__iter__'):
            for t in tblname:
                self.policy.append( (t, handler) )      # append to policy
        else:
            raise TypeError('Table-name must be a string (or a list of strings)')
        return handler

    def remove_handler(self, handler):
        """ Remove specified handler so that data is no longer stored there. """

        if isinstance(handler, DataHandler):
            for a_tblname, a_handler in self.policy[:]:
                if a_handler == handler:
                    self.policy.remove((a_tblname, a_handler))
            handler.close()
            self._lookup_cache = {}
        else:
            raise ValueError("Please provide valid DataHandler object.")

    @only_root
    def close(self):
        """ Reset the datalog and close all registered DataHandlers """
        for (tblname, handler) in self.policy:
            handler.close()

#=============================================================================
# Create global default data logger

#~ dlog = DataLog()
