import numpy as np
import os

class MPIBase(object):
    '''
    Parent class for MPI related stuff
    '''

    def __init__(self, mpi=True, comm=None, **kwargs):
        '''
        Check if MPI is working by checking common
        MPI environment variables and set MPI atrributes.

        Keyword arguments
        ---------
        mpi : bool
            If False, do not use MPI regardless of MPI env.
            otherwise, let code decide based on env. vars
            (default : True)
        comm : MPI.comm object, None
            External communicator. If left None, create
            communicator. (default : None)
        '''

        super(MPIBase, self).__init__(**kwargs)

        # Check whether MPI is working
        # add your own environment variable if needed
        # Open MPI environment variable
        ompi_size = os.getenv('OMPI_COMM_WORLD_SIZE')
        # intel and/or mpich environment variable
        pmi_size = os.getenv('PMI_SIZE')

        if not (ompi_size or pmi_size) or not mpi:
            self.mpi = False

        else:
            try:
                from mpi4py import MPI

                self.mpi = True
                self._mpi_double = MPI.DOUBLE
                self._mpi_sum = MPI.SUM
                if comm:
                    self._comm = comm
                else:
                    self._comm = MPI.COMM_WORLD

            except ImportError:
                warn("Failed to import mpi4py, continuing without MPI",
                     RuntimeWarning)

                self.mpi = False

    @property
    def mpi_rank(self):
        if self.mpi:
            return self._comm.Get_rank()
        else:
            return 0

    @property
    def mpi_size(self):
        if self.mpi:
            return self._comm.Get_size()
        else:
            return 1

    def barrier(self):
        '''
        MPI barrier that does nothing if not MPI
        '''
        if not self.mpi:
            return
        self._comm.Barrier()

    def scatter_list(self, list_tot, root=0):
        '''
        Scatter python list from `root` even if
        list is not evenly divisible among ranks.

        Arguments
        ---------
        list_tot : array-like
            List or array to be scattered (in 0-axis).
            Not-None on rank specified by `root`.

        Keyword arguments
        -----------------
        root : int
            Root rank (default : 0)
        '''

        if not self.mpi:
            return list_tot

        if self.mpi_rank == root:
            arr = np.asarray(list_tot)
            arrs = np.array_split(arr, self.mpi_size)

        else:
            arrs = None

        arrs = self._comm.scatter(arrs, root=root)

        return arrs.tolist()

    def broadcast(self, obj):
        '''
        Broadcast a python object that is non-None on root
        to all other ranks. Can be None on other ranks, but
        should exist in scope.

        Arguments
        ---------
        obj : object

        Returns
        -------
        bcast_obj : object
            Input obj, but now on all ranks
        '''

        if not self.mpi:
            return obj

        obj = self._comm.bcast(obj, root=0)
        return obj

    def broadcast_array(self, arr):
        '''
        Broadcast array from root process to all other ranks.

        Arguments
        ---------
        arr : array-like or None
            Array to be broadcasted. Not-None on root
            process, can be None on other ranks.

        Returns
        -------
        bcast_arr : array-like
            input array (arr) on all ranks.
        '''

        if not self.mpi:
            return arr

        # Broadcast meta info first
        if self.mpi_rank == 0:
            shape = arr.shape
            dtype = arr.dtype
        else:
            shape = None
            dtype = None
        shape, dtype = self._comm.bcast((shape, dtype), root=0)

        if self.mpi_rank == 0:
            bcast_arr = arr
        else:
            bcast_arr = np.empty(shape, dtype=dtype)

        self._comm.Bcast(bcast_arr, root=0)

        return bcast_arr

    def reduce_array(self, arr_loc):
        '''
        Sum arrays on all ranks elementwise into an
        array living in the root process.

        Arguments
        ---------
        arr_loc : array-like
            Local numpy array on each rank to be reduced.
            Need to be of same shape and dtype on each rank.

        Returns
        -------
        arr : array-like or None
            Reduced numpy array with same shape and dtype as
            arr_loc on root process, None for other ranks
            (arr_loc if not using MPI)
        '''

        if not self.mpi:
            return arr_loc

        if self.mpi_rank == 0:
            arr = np.empty_like(arr_loc)
        else:
            arr = None

        self._comm.Reduce(arr_loc, arr, op=self._mpi_sum, root=0)

        return arr

    def distribute_array(self, arr):
        '''
        If MPI is enabled, give every rank a proportionate
        view of the total array (or list).

        Arguments
        ---------
        arr : array-like, list
            Full-sized array present on every rank

        Returns
        -------
        arr_loc : array-like
            View of array unique to every rank.
        '''

        if self.mpi:

            sub_size = np.zeros(self.mpi_size, dtype=int)
            quot, remainder = divmod(len(arr), self.mpi_size)
            sub_size += quot

            if remainder:
                # give first ranks extra element
                sub_size[:int(remainder)] += 1

            start = np.sum(sub_size[:self.mpi_rank], dtype=int)
            end = start + sub_size[self.mpi_rank]

            arr_loc = arr[start:end]

        else:
            arr_loc = arr

        return arr_loc
