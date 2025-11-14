import debugpy
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Debugpy listens on a unique port for each rank
port = 5678 + rank
debugpy.listen(("localhost", port))

print(f"Rank {rank}: Waiting for debugger to attach on port {port}", flush=True)
debugpy.wait_for_client()
debugpy.breakpoint()

print(f"Rank {rank}: Debugger attached!", flush=True)

# Add a barrier to synchronize all processes
comm.Barrier()

if size >= 2:
    if rank == 0:
        # Rank 0 sends data to rank 1
        data_to_send = np.array([1.0, 2.0, 3.0, 4.0])
        comm.send(data_to_send, dest=1, tag=11)
        print(f"Rank 0: Sent data to rank 1: {data_to_send}", flush=True)

    elif rank == 1:
        # Rank 1 receives data from rank 0
        received_data = comm.recv(source=0, tag=11)
        print(f"Rank 1: Received data from rank 0: {received_data}", flush=True)

# Add another barrier to ensure all processes finish the exchange before exiting
comm.Barrier()

print(f"Rank {rank}: Execution finished.", flush=True)
