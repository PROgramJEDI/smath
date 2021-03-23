import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.misc import derivative



def set_axes():
	fig, ax = plt.subplots()
	ax.grid(True, which='both')

	ax.axhline(y=0, color='black')
	ax.axvline(x=0, color='black')


def relfunc(f, rel, x_start=-20, x_end=20, accuracy=10, dx=10**(-3)):
	# the first derivitive of f(x).
	f_tag = lambda x: derivative(f, x, dx=dx)
	f_arc_length = lambda x: np.sqrt(1 + np.square(f_tag(x)))

	a_range = np.linspace(x_start, x_end, (x_end - x_start) * accuracy)
	x_arc_length = np.array([quad(f_arc_length, 0, x)[0] for x in a_range])

	# the orthagonal line to the tangent on a specific point.
	m = lambda x: (-1/f_tag(x_arc_length))*(x-x_arc_length)+f(x_arc_length)
	solutions_func = lambda x: m(x) - rel(x)
	# solve rel(x) = m(x).
	x_solutions = np.asarray(fsolve(solutions_func, x_arc_length))

	y_diff = m(x_solutions) - m(x_arc_length)
	x_diff = x_solutions - x_arc_length

	distances = np.sqrt(x_diff ** 2 + y_diff ** 2) * (y_diff / abs(y_diff))
	return zip(x_arc_length, distances)
