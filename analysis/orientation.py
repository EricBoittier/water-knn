import numpy as np
import MDAnalysis as mda
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exponential_func(x, A, B):
    """
    Exponential function
    A: Amplitude
    B: Time constant
    """
    return (A * np.exp(-(x-1) /B)) 


from scipy.special import legendre
l2 = legendre(2)
l1 = legendre(1)

from pathlib import Path

def load_universe(psf, dcd):
    import MDAnalysis as mda
    u = mda.Universe(psf, dcd)
    return u

def get_orientation_correlation(u, Values=[], Values2=[], 
    repeat=1, 
    fn="",
    legendre_order=2, N=1000, NSAVC=100, NMOL=4000, DT=0.0002, 
    NSKIP = 1,
    NSKIP2 = -1):
    """
    Get the orientation correlation of the water molecules
    """
    # Get the legendre polynomial
    if legendre_order == 2:
        LEGENDRE = l2
    elif legendre_order == 1:
        LEGENDRE = l1
    else:
        raise ValueError(f"Legendre order {legendre_order} not supported")

    all_vectors = []
    all_times = []
    # Get the indices of the atoms
    index_O1s = u.select_atoms("type OT").indices
    index_H1s = u.select_atoms("type HT").indices[1::2]
    index_H2s = u.select_atoms("type HT").indices[::2]
    # Loop over the trajectory
    for t, _ in enumerate(u.trajectory[N*repeat:N*repeat+N]):
        atoms = u.select_atoms("all")
        all_times.append(t * DT * NSAVC)
        vectors = []
        pos_array = np.array(atoms.positions)
        # Get the positions of the atoms and the bond vectors
        O1s = pos_array[index_O1s]
        H1s = pos_array[index_H1s]
        H2s = pos_array[index_H2s]
        v1_norm = np.linalg.norm(O1s - H1s, axis=-1)
        v2_norm = np.linalg.norm(O1s - H2s, axis=-1)
        unit1 = (O1s - H1s) / v1_norm[:, np.newaxis]
        unit2 = (O1s - H2s) / v2_norm[:, np.newaxis]
        # Get the vectors
        vectors = np.concatenate([unit1, unit2])
        all_vectors.append(vectors)
    all_vectors = np.array(all_vectors)
    
    # Get the angles    
    all_angles = []
    for j in range(NMOL):
        angles = []
        for i in range(N):
            a = np.dot(all_vectors[0][j], all_vectors[i][j])
            angles.append(LEGENDRE(a))
        all_angles.append(angles)

    res = np.array(all_angles).T.mean(axis=1)
    x_data = np.array([i * DT * NSAVC for i in range(N)])
    y_data = res

    bounds = ([0., 0], [np.inf, np.inf])
    params, cov_matrix = curve_fit(exponential_func, 
                                   x_data[NSKIP:NSKIP2], 
                                   y_data[NSKIP:NSKIP2], 
                                   bounds=bounds, 
                                   maxfev=10000)
    A_fit, B_fit = params
    A_err, B_err = np.sqrt(np.diag(cov_matrix))

    x_fit = np.linspace(min(x_data), max(x_data), 100) 
    y_fit = exponential_func(x_fit, A_fit, B_fit)

    Values.append(B_fit)
    Values2.append(A_fit)
    print("A, B:", A_fit, B_fit)
    print("A_err, B_err:", A_err, B_err)

    # plt.plot(np.array([i * 0.0002 * 100 for i in range(1000)])[NSKIP:NSKIP2], res[NSKIP:NSKIP2], "-*")
    plt.scatter(x_data, y_data, s=0.1, alpha=0.1, c="r")
    plt.ylim(0,1)
    plt.xlim(0,20)
    plt.plot(x_fit, y_fit, alpha=0.6)
    plt.savefig(f"{fn}-orientation_correlation-{repeat}.png")

    return Values, Values2


def check_args(args):
    if args.REPEAT < 1:
        raise ValueError("REPEAT must be greater than 0")
    if args.NSKIP < 0:
        raise ValueError("NSKIP must be greater than 0")
    # if args.NSKIP2 < 0:
    #     raise ValueError("NSKIP2 must be greater than 0")
    if not Path(args.dcd).exists():
        raise ValueError("dcd file does not exist")
    if not Path(args.psf).exists():
        raise ValueError("psf file does not exist")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--psf", type=str, required=True)
    parser.add_argument("--dcd", type=str, required=True)
    parser.add_argument("--legendre_order", type=int, default=2)
    parser.add_argument("--N", type=int, default=1000)
    parser.add_argument("--NSAVC", type=int, default=100)
    parser.add_argument("--REPEAT", type=int, default=1)
    parser.add_argument("--NMOL", type=int, default=4000)
    parser.add_argument("--DT", type=float, default=0.0002)
    parser.add_argument("--NSKIP", type=int, default=10)
    parser.add_argument("--NSKIP2", type=int, default=-1)
    
    args = parser.parse_args()
    check_args(args)

    u = load_universe(args.psf, args.dcd)
    print("Total frames:", len(u.trajectory))
    fn = args.dcd.split("/")[-3]
    print("Processing", fn)
    Values, Values2 = [], []

    for i in range(args.REPEAT):
        print(f"Processing {i}th frame")
        Values, Values2 = get_orientation_correlation(u, Values, Values2, i, 
            legendre_order=args.legendre_order, N=args.N, NSAVC=args.NSAVC, 
            NMOL=args.NMOL, DT=args.DT, NSKIP=args.NSKIP, NSKIP2=args.NSKIP2, fn=fn)
    
    Values = np.array(Values)
    Values2 = np.array(Values2)
    print(Values.mean(), Values2.mean())
    print(Values.std(), Values2.std())
    open("orientation_correlation.csv", "a").write(f"{fn}, {Values.mean()}, {Values.std()}\n")
    open("orientation_correlation2.csv", "a").write(f"{fn}, {Values2.mean()}, {Values2.std()}\n")
    plt.clf()
    plt.scatter(Values, Values2, s=0.1, alpha=1, c="r")
    plt.savefig("orientation_correlation.png")

if __name__ == "__main__":
    main()
