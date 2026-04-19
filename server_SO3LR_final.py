#!/usr/bin/env python3
from __future__ import print_function
import jax
import jax.numpy as jnp
from jax_md import space, quantity
from so3lr import So3lrPotential, to_jax_md
import numpy as np
import socket
import os
import sys
import time
import traceback

# =====================
# CONFIGURATION
# =====================
sock_address = os.environ.get("SO3LR_SOCKET", "/scratch/v15/hp2306/so3lr_socket")
EV_TO_KCAL = 23.0609

print(f"JAX devices: {jax.devices()}", flush=True)

ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'S': 16, 'Cl': 17, 'P': 15
}

# Remove old socket if exists
if os.path.exists(sock_address):
    print(f"Removing old socket: {sock_address}", flush=True)
    os.remove(sock_address)

# =====================
# FUNCTION TO READ INPUT
# =====================
def readinput(fname):
    """Reads input file from NAMD client."""
    with open(fname, "r") as infile:
        first_line = infile.readline().strip()
        fields = first_line.split()
        if len(fields) < 2:
            raise ValueError(f"First line malformed: '{first_line}'")
        numQMatms = int(fields[0])
        numMMatms = int(fields[1])

        elements = []
        crd = []
        charges = []
        lineIndx = 1
        for line in infile:
            stripped = line.strip()
            if not stripped:
                lineIndx += 1
                continue
            f = stripped.split()
            if lineIndx <= numQMatms:
                x, y, z = float(f[0]), float(f[1]), float(f[2])
                elem = f[3]
                elements.append(elem)
                crd.append([x, y, z])
            else:
                charge = float(f[3]) if len(f) >= 4 else 0.0
                charges.append(charge)
            lineIndx += 1
    return elements, crd, charges

# =====================
# INITIALISE SO3LR
# =====================
potential      = None
energy_fn      = None
force_fn       = None
neighbor_fn    = None
neighbor_fn_lr = None
nbrs           = None
nbrs_lr        = None
prev_n_atoms   = None

# =====================
# START SOCKET SERVER
# =====================
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sock_address)
server.listen(100)
print(f"Server listening at socket: {sock_address}", flush=True)

try:
    while True:
        print("DEBUG: Waiting for connection...", flush=True)
        conn, _ = server.accept()
        conn.settimeout(10.0)
        print("DEBUG: Connection accepted.", flush=True)

        try:
            datagram = conn.recv(1024)
            if not datagram:
                continue
            fname = datagram.decode('utf-8').strip()
            print(f"DEBUG: Received filename: '{fname}'", flush=True)
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Input file not found: {fname}")

            # Read input
            elements, crd, charges = readinput(fname)
            n_atoms = len(elements)
            print(f"DEBUG: Preparing JAX arrays on GPU...", flush=True)

            # Move arrays explicitly to GPU
            atomic_nums = jax.device_put(jnp.array([ATOMIC_NUMBERS[e] for e in elements], dtype=jnp.int32))
            positions   = jax.device_put(jnp.array(crd, dtype=jnp.float32))

            # Initialise SO3LR if first call or atom count changed
            if potential is None or n_atoms != prev_n_atoms:
                print(f"DEBUG: Initialising SO3LR for {n_atoms} atoms...", flush=True)
                potential = So3lrPotential()
                displacement, _ = space.free()
                neighbor_fn, neighbor_fn_lr, raw_energy_fn = to_jax_md(
                    potential=potential,
                    displacement_or_metric=displacement,
                    box_size=None,
                    species=atomic_nums,
                    capacity_multiplier=1.25,
                    buffer_size_multiplier_sr=1.25,
                    buffer_size_multiplier_lr=1.25,
                    minimum_cell_size_multiplier_sr=1.0,
                    disable_cell_list=True,
                    fractional_coordinates=False,
                )
                energy_fn = jax.jit(raw_energy_fn)
                force_fn  = jax.jit(quantity.force(raw_energy_fn))
                nbrs      = neighbor_fn.allocate(positions, box=None)
                nbrs_lr   = neighbor_fn_lr.allocate(positions, box=None)
                prev_n_atoms = n_atoms
                print("DEBUG: SO3LR initialized and JIT compiled.", flush=True)

            # Update neighbor lists
            nbrs    = nbrs.update(positions, box=None)
            nbrs_lr = nbrs_lr.update(positions, box=None)
            if nbrs.did_buffer_overflow or nbrs_lr.did_buffer_overflow:
                nbrs    = neighbor_fn.allocate(positions, box=None)
                nbrs_lr = neighbor_fn_lr.allocate(positions, box=None)

            # =====================
            # Compute energy and forces
            # =====================
            energy = energy_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=None)
            forces = force_fn(positions, neighbor=nbrs.idx, neighbor_lr=nbrs_lr.idx, box=None)
            jax.block_until_ready(forces)

            # Write results
            result_file = fname + '.result'
            print(f"DEBUG: Writing results to {result_file}", flush=True)
            d = np.array(forces)
            with open(result_file, 'w') as finFile:
                finFile.write(f"{float(energy) * EV_TO_KCAL}\n")
                for i in range(n_atoms):
                    fx, fy, fz = EV_TO_KCAL * d[i]
                    finFile.write(f"{fx} {fy} {fz} 0.0\n")
            print("DEBUG: Results written successfully.", flush=True)

            # Send done signal to client
            conn.send("D\r".encode("utf-8"))
            print("DEBUG: Sent done signal to client.", flush=True)

        except Exception as e:
            print(f"ERROR processing request: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            try:
                conn.send("E\r".encode("utf-8"))
            except Exception:
                pass

        finally:
            conn.close()
            print("DEBUG: Connection closed.", flush=True)

except KeyboardInterrupt:
    print("Server interrupted by user. Closing...", flush=True)

finally:
    server.close()
    if os.path.exists(sock_address):
        os.remove(sock_address)
    print("Server shutdown. Socket removed.", flush=True)