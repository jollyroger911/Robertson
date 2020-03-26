import numpy as np
import math as m
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

import solvers


def first_equation(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)

	u[0] = -1.3e-3 * u2 - 1e3 * u1 * u2 - 2.5e3 * u1 * u3
	u[1] = -1.3e-3 * u2 - 1e3 * u1 * u2
	u[2] =                              - 2.5e3 * u1 * u3

	return u

def fe_jacob(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = -1e3 * u2 - 2.5e3 * u3; res[0][1] = -1.3e-3 * u2 - 1e3 * u1; res[0][2] = -2.5e3 * u1;
	res[1][0] = -1e3 * u2;              res[1][1] = -1.3e-3 * u2 - 1e3 * u1; res[1][2] = 0.0;
	res[2][0] =           - 2.5e3 * u3; res[2][1] = 0.0;                     res[2][2] = -2.5e3 * u1;
	
	return res
	

def second_equation(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)

	u[0] = -0.04 * u1 + 1e4 * u2 * u3
	u[1] = +0.04 * u1 - 1e4 * u2 * u3 - 3e7 * u2 * u2
	u[2] =                            + 3e7 * u2 * u2
	
	return u

def se_jacob(t, u):
	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = -0.04 + 1e4 * u3; res[0][1] = +1e4 * u3               ; res[0][2] = +1e4 * u2;
	res[1][0] = +0.04 - 1e4 * u3; res[1][1] = -1e4 * u3 - 3e7 * 2 * u2; res[1][2] = -1e4 * u2;
	res[2][0] = 0;                res[2][1] =           + 3e7 * 2 * u2; res[2][2] = 0;
	
	return res


def third_equation(t, u):
	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u1, u2, u3 = u[0], u[1], u[2]

	u = np.zeros(u.shape, np.float64)
	u[0] = a * u2 * u3
	u[1] = b * u3 * u1
	u[2] = c * u1 * u2	
	return u

def te_jacob(t, u):
	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u1, u2, u3 = u[0], u[1], u[2]

	res = np.zeros((3,3), np.float64)
	res[0][0] = 0;      res[0][1] = a * u3; res[0][2] = a * u2;
	res[1][0] = b * u3; res[1][1] = 0;      res[1][2] = b * u1;
	res[2][0] = c * u2; res[2][1] = c * u1; res[2][2] = 0;
	return res


def integrate_equation(solver, dt, tn, output_every=1000):
	i = 0
	while True:
		if i % output_every == 0:
			t, u = solver.get_state()
			print(f't: {t} | u: {u} |')
		i += 1

		t = solver.t
		if t < tn:
			solver.evolve(t, dt)
		else:
			break


def solve_first():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((0, 1, 1))                          # initial value
	tn = 500.0                                          # final time
	un = np.float64((-1.893e-7, 0.5976547, 1.40223434)) # final value(for test)
	n = 5000                                            # count of intervals
	dt = (tn - t0) / n                                  # time delta

	f = first_equation
	j = fe_jacob

	# Initialize your solver here and call integrate_equation()
	solver = solvers.RKI_naive(solvers.lobattoIIIC_2(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)

	# ini_solver = solvers.RKI_naive(solvers.gauss_legendre_6(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)

	# a, b = solvers.build_implicit_adams(2)
	# solver = solvers.ImplicitMultistepSolver(ini_solver, solvers.NeutonSolver(1e-15, 100), a, b, dt)

	integrate_equation(solver, dt, tn)

	yn = solver.value()

	# Results
	print('---First equation. Expected and got result---')
	print(f'Expected:{un}')
	print(f'Got     :{yn}')

def solve_second():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((1, 0, 0))                          # initial value
	tn = 40.0                                           # final time
	un = np.float64((0.7158271, 9.186e-6, 0.2841637))   # final value(for test)
	n = 80000000                                        # count of intervals
	dt = (tn - t0) / n                                  # time delta

	f = second_equation
	j = se_jacob

	# Initialize your solver here and call integrate_equation()
	# solver = solvers.RKI_naive(solvers.gauss_legendre_6(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)
	solver = solvers.RKE(solvers.classic_4(), f, t0, u0)

	integrate_equation(solver, dt, tn)

	yn = solver.value()

	# Results
	print('---Second equation. Expected and got result---')
	print(f'Expected:{un}')
	print(f'Got     :{yn}')

def solve_third():
	# initial value problem
	t0 = 0.0                                            # initial time
	u0 = np.float64((m.cos(1.1), 0, m.sin(1.1)))        # initial value
	tn = 100.0                                          # final time
	n = 1000										    # count of intervals
	dt = (tn - t0) / n                                  # time delta
	output_every = 1

	I1 = 2
	I2 = 1
	I3 = 2/3
	a = (I2 - I3) / (I2 * I3)
	b = (I3 - I1) / (I3 * I1)
	c = (I1 - I2) / (I1 * I2)

	u01, u02, u03 = u0
	i1 = u01 * u01 + u02 * u02 + u03 * u03
	i2 = u01 * u01 / I1 + u02 * u02 / I2 + u03 * u03 / I3

	f = third_equation
	j = te_jacob

	def integrate_with_solver(solver):
		i = 0
		t_values = []
		u1_values = []
		u2_values = []
		u3_values = []
		inv_first  = []
		inv_second = []
		while True:
			if i % output_every == 0:
				t, u = solver.get_state()
				u1, u2, u3 = u

				t_values.append(t)
				u1_values.append(u1)
				u2_values.append(u2)
				u3_values.append(u3)

				inv_first.append(u1 * u1 + u2 * u2 + u3 * u3)
				inv_second.append(u1 * u1 / I1 + u2 * u2 / I2 + u3 * u3 / I3)
			i += 1

			t = solver.t
			if t < tn:
				solver.evolve(t, dt)
			else:
				break

		max_delta = 0.0
		for inv in inv_first:
			max_delta = max(max_delta, m.fabs(i1 - inv))
		print('Maximum first invariant deviation: ', max_delta)

		max_delta = 0.0
		for inv in inv_second:
			max_delta = max(max_delta, m.fabs(i2 - inv))
		print('Maximum second invariant deviation: ', max_delta)

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		#
		# # Make data
		# u = np.linspace(0, 2 * np.pi, 100)
		# v = np.linspace(0, np.pi, 100)
		# x = 10 * np.outer(np.cos(u), np.sin(v))
		# y = 10 * np.outer(np.sin(u), np.sin(v))
		# z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

		# Plot the surface
		# fig1 = pyplot.figure()
		# fig1
		# ax.plot_surface(x, y, z, color='b')
		# pyplot.show()
		#
		# fig = pyplot.figure()
		# ax = fig.add_subplot(1, 1, 1, projection='3d')
		# ax = fig.add_subplot(x, y, z, color='b')
		ax.scatter(u1_values, u2_values, u3_values)
		# ax.scatter(x, y, z, color='b')
		pyplot.show()

	solver = solvers.RKI_naive(solvers.implicit_midpoint_2(), f, j, solvers.NeutonSolver(1e-15, 100), t0, u0)
	integrate_with_solver(solver)

	solver = solvers.RKE(solvers.classic_4(), f, t0, u0)
	integrate_with_solver(solver)
	
	solver = solvers.RKE(solvers.explicit_euler_1(), f, t0, u0)
	integrate_with_solver(solver)
	

def main():
	# solve_first()
	# solve_second()
	solve_third()


if __name__ == '__main__':
	main()

	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()
	print('Press any key...', end='')
	input()