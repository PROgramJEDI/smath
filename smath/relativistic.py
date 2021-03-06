import numpy as np

from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.integrate import cumulative_trapezoid as integral



def plot(f, rel, accuracy=10, x_start=-20, x_end=20, dx=1e-4):
	import matplotlib.pyplot as plt

	fig, ax = plt.subplots()
	ax.grid(True, which='both')
	ax.axhline(y=0, color='black')
	ax.axvline(x=0, color='black')

	x, y = relfunc(f, rel, accuracy, x_start, x_end, dx)
	plt.plot(x, y)
	plt.show()


def relfunc(f, rel, accuracy=10, x_start=-20, x_end=20, dx=1e-4):
	# the first derivitive of f(x).
	f_tag = lambda x: derivative(f, x, dx=dx)
	f_arc_length = lambda x: np.sqrt(1 + np.square(f_tag(x)))

	# the x-axis of the current system of reference.
	a_range = np.linspace(x_start, x_end, int((x_end - x_start) * accuracy))
	a_range = a_range[a_range != 0]

	# prepare the two groups to be integrated separately.
	a_negatives, a_positives = a_range[a_range < 0], a_range[a_range > 0] 

	# the continuous integration of the two groups.
	negetive_x_arc_length = integral(f_arc_length(a_negatives), a_negatives, dx=dx)[::-1]	
	positive_x_arc_length = integral(f_arc_length(a_positives), a_positives, dx=dx, initial=0)

	# reduce to the same shape of the axis of the negatives integration.
	a_range = a_range[1:]
	# concat the groups into one field. the negitive group should be negitive.
	x_arc_length = np.concatenate((-negetive_x_arc_length, positive_x_arc_length))
	
	# applying the given range for the new reference system.
	x_arc_range = (x_arc_length >= x_start) & (x_arc_length <= x_end)
	x_arc_length = x_arc_length[x_arc_range]

	# the orthogonal linear function to the tangent in a specific point.
	m = lambda x: (-1/f_tag(x_arc_length))*(x-x_arc_length)+f(x_arc_length)
	solutions_func = lambda x: m(x) - rel(x)
	
	# solve rel(x) = m(x).
	x_solutions = fsolve(solutions_func, x_arc_length)

	y_diff = m(x_solutions) - m(x_arc_length)
	x_diff = x_solutions - x_arc_length

	distances = np.sqrt(np.square(x_diff) + np.square(y_diff)) * (y_diff / np.absolute(y_diff))
	return x_arc_length, distances
