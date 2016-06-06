#seperateCodes.py
from mpi4py import MPI
a = 6.0
b = 3.0
print ("Two numbers provided ", a, b)
rank = MPI.COMM_WORLD.Get_rank()


if rank == 0:
        print (a + b)
        print ("Addition from proces", rank)
if rank == 1:
        print (a * b)
        print ("Multiplies from proces", rank)
if rank == 2:
        print (max(a,b))
        print ("Maxium number  from proces", rank)
