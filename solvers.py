import math as m
import numpy as np
import numpy.polynomial as poly
import numpy.linalg as lin


# utility
def factorial(num):
	res = 1
	while num > 1:
		res *= num
		num -= 1
	return res

def kronecker_delta(i, j):

	return 1 if i == j else 0

			
class DiffJacobian1:
	def __init__(self, func, eps = 1e-6):
		self.func = func
		self.eps  = eps

	def __call__(self, u):
		order = len(u)
		jac = np.zeros((order, order), np.float64)

		um1 = u.copy()
		up1 = u.copy()
		#iterate through variables
		for j in range(order):
			#assign u
			for i in range(order):
				um1[i] = u[i]
			for i in range(order):
				up1[i] = u[i]

			#add/subtract eps from i-th component
			um1[j] -= self.eps
			up1[j] += self.eps

			fm1 = self.func(um1)
			fp1 = self.func(up1)

			#fill j-th column
			for i in range(order):
				jac[i][j] = (fp1[i] - fm1[i]) / (2 * self.eps)

		return jac

class DiffJacobian2:
	def __init__(self, func, eps = 1e-6):
		self.func = func
		self.eps  = eps

	def __call__(self, t, u):
		order = len(u)
		jac = np.zeros((order, order), np.float64)

		um1 = u.copy()
		up1 = u.copy()
		#iterate through variables
		for j in range(order):
			#assign u
			for i in range(order):
				um1[i] = u[i]
			for i in range(order):
				up1[i] = u[i]

			#add/subtract eps from i-th component
			um1[j] -= self.eps
			up1[j] += self.eps

			fm1 = self.func(t, um1)
			fp1 = self.func(t, up1)

			#fill j-th column
			for i in range(order):
				jac[i][j] = (fp1[i] - fm1[i]) / (2 * self.eps)

		return jac



#solver for nonlinear systems
class NeutonSolver:
	def __init__(self, eps, iter_lim):
		self.eps = eps
		self.iter_lim = iter_lim

	def solve(self, func, jacob, x0):
		x_k = x0.copy()
		iters = 0

		while True:
			sys  = jacob(x_k)
			term = -func(x_k)

			delta = lin.solve(sys, term)

			x_k += delta
			iters += 1

			if abs(delta).max() < self.eps or iters >= self.iter_lim:
				break

		return x_k


# solver base class
class SolverBase:
	def __init__(self, function, jacobian, t0, u0):
		self.function = function
		self.jacobian = jacobian

		self.t = t0
		self.u = u0.copy()

	def set_state(self, t, u):
		self.t = t
		self.u = u

	def get_state(self):
		return self.t, self.u.copy()

	def value(self):
		return self.u
	
	def evolve(self, t, dt):
		# to override
		pass
	

# Runge-Kutta methods
class Tableau:
	def __init__(self, order, aMat, bVec, cVec):
		self.order = order
		self.aMat = np.copy(aMat)
		self.bVec = np.copy(bVec)
		self.cVec = np.copy(cVec)

# explicit
def explicit_euler_1():
	a_mat = np.float64( ((0,),) )
	b_vec = np.float64( (1,) )
	c_vec = np.float64( (0,) )

	return Tableau(1, a_mat, b_vec, c_vec)

def classic_4():
	aMat = np.float64(
		(
			  (0.0, 0.0, 0.0, 0.0)
			, (0.5, 0.0, 0.0, 0.0)
			, (0.0, 0.5, 0.0, 0.0)
			, (0.0, 0.0, 1.0, 0.0)
		)
	)

	bVec = np.float64((1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0))

	cVec = np.float64((0.0, 0.5, 0.5, 1.0))

	return Tableau(4, aMat, bVec, cVec)

# implicit
def backward_euler_1():
	a_mat = np.float64( ((1,),) )
	b_vec = np.float64( (1,) )
	c_vec = np.float64( (1,) )

	return Tableau(1, a_mat, b_vec, c_vec)

def gauss_2():
	a_mat = np.float64( ((1/2,),) )
	b_vec = np.float64( (1,) )
	c_vec = np.float64( (1/2,) )

	return Tableau(1, a_mat, b_vec, c_vec)

def implicit_midpoint_2():
	a_mat = np.float64(((0.5,),))
	b_vec = np.float64((1.0,))
	c_vec = np.float64((0.5,))

	return Tableau(1, a_mat, b_vec, c_vec)

def lobattoIIIC_2():
	a_mat = np.float64((
		  (1/2, -1/2)
		, (1/2,  1/2) 
	))
	b_vec = np.float64((1/2, 1/2))
	c_vec = np.float64(( 0 ,  1 ))

	return Tableau(2, a_mat, b_vec, c_vec)

def lobattoIIIC_4():
	a_mat = np.float64((
		  (1/6, -1/3,  1/6 )
		, (1/6, 5/12, -1/12) 
		, (1/6,  2/3,  1/6 )
	))
	b_vec = np.float64((1/6, 2/3, 1/6))
	c_vec = np.float64((0  , 1/2,   1))

	return Tableau(3, a_mat, b_vec, c_vec)

def gauss_legendre_6():
	sq15 = m.sqrt(15.0)
	
	a_mat = np.float64((
		  (5/36          , 2/9 - sq15/15, 5/36 - sq15/30)
		, (5/36 + sq15/24, 2/9          , 5/36 - sq15/24) 
		, (5/36 + sq15/30, 2/9 + sq15/15, 5/36          )
	))
	b_vec = np.float64((5/18, 4/9, 5/18))
	c_vec = np.float64((1/2 - sq15/10, 1/2, 1/2 + sq15/10))

	return Tableau(3, a_mat, b_vec, c_vec)


class RKE(SolverBase):
	def __init__(self, tableau, func, t0, u0):
		super(RKE, self).__init__(func, None, t0, u0) 

		self.tableau = tableau
		self.rke_order = tableau.order
		self.sys_order = len(u0)

	def evolve(self, t, dt):
		aMat, bVec, cVec = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		kVecs = np.zeros((self.rke_order, self.sys_order), np.float64)
		for i in range(self.rke_order):
			for j in range(i):
				kVecs[i] += dt * kVecs[j] * aMat[i][j]
			kVecs[i] = self.function(t + dt * cVec[i], self.u + kVecs[i])

		du = np.zeros((self.sys_order, ), np.float64)
		for i in range(self.rke_order):
			du += dt * bVec[i] * kVecs[i]

		self.t = t + dt # DANGER
		self.u += du

class RKI_naive(SolverBase):
	def __init__(self, tableau, function, jacobian, neuton_solver, t0, u0):
		super(RKI_naive, self).__init__(function, jacobian, t0, u0)

		self.tableau = tableau
		self.rki_order = tableau.order
		self.sys_order = len(u0)

		self.dt = 0.0

		self.neuton_solver = neuton_solver	

	def evolve(self, t, dt):
		self.t  = t
		self.dt = dt

		u = self.u

		N = self.sys_order
		s = self.rki_order

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec
		
		#initial guess
		value = self.function(t, u)
		z = np.zeros((s, N), np.float64)
		for i in range(s):
			z[i] = value
		z = z.reshape((N * s))

		#solving system
		z = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, z)

		#computing delta
		z = z.reshape((s, N))

		du = np.zeros((N,), np.float64)
		for i in range(s):
			du += b[i] * z[i]
		du *= dt

		#update result
		self.t = t + dt # DANGER
		self.u += du

	def __special_jacobian(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_jacobian = np.zeros((N * s, N * s), np.float64)
		for i in range(s):
			sums = np.zeros((N,), np.float64)
			for j in range(s):
				sums += a[i][j] * z[j]
			sums *= dt

			jacobian = self.jacobian(t + dt * c[i], u + sums)
			for I in range(N):
				for j in range(s):
					for J in range(N):
						dij = kronecker_delta(i, j)
						dIJ = kronecker_delta(I, J)

						elem = dij * dIJ - dt * a[i][j] * jacobian[I][J]

						special_jacobian[N * i + I][N * j + J] = elem

		return special_jacobian

	def __special_function(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))
		
		special_function = np.zeros((s, N), np.float64)
		for i in range(s):
			sums = np.zeros((N,), np.float64)
			for j in range(s):
				sums += a[i][j] * z[j]
			sums *= dt

			special_function[i] = self.function(t + dt * c[i], u + sums)
				
		z = z.reshape((N * s))
		special_function = special_function.reshape((N * s))

		return z - special_function		

#TEST
class RKI_better(SolverBase):
	def __init__(self, tableau, function, jacobian, neuton_solver, t0, u0):
		super(RKI_better, self).__init__(function, jacobian, t0, u0)

		self.tableau = tableau
		self.rki_order = tableau.order
		self.sys_order = len(u0)

		self.dt = 0.0

		self.neuton_solver = neuton_solver	

	def evolve(self, t, dt):
		self.t  = t
		self.dt = dt

		u = self.u

		N = self.sys_order
		s = self.rki_order

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec
		
		#initial guess
		z = np.zeros((N * s), np.float64)

		#solving system
		z = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, z)

		#computing delta
		z = z.reshape((s, N))
		
		du = np.zeros((N,), np.float64)
		for i in range(s):
			du += b[i] * self.function(t + dt * c[i], u + z[i])
		du *= dt

		#update result
		self.t = t + dt
		self.u += du

	def __special_jacobian(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_jacobian = np.zeros((s, N, s, N), np.float64)
		for j in range(s):
			jacobian = self.jacobian(t + c[j] * dt, u + z[j])

			for i in range(s):
				dij = kronecker_delta(i, j)

				for I in range(N):

					for J in range(N):
						dIJ = kronecker_delta(I, J)

						special_jacobian[i][I][j][J] = dij * dIJ - dt * a[i][j] * jacobian[I][J]

		return special_jacobian.reshape((N * s, N * s))

	def __special_function(self, z):
		N = self.sys_order	
		s = self.rki_order

		u, t, dt = self.u, self.t, self.dt

		a, b, c = self.tableau.aMat, self.tableau.bVec, self.tableau.cVec

		z = z.reshape((s, N))

		special_function = np.zeros((s, N), np.float64)
		special_function[:,:] = z[:,:]
		
		for j in range(s):
			function = self.function(t + c[j] * dt, u + z[j])
			for i in range(s):
				special_function[i] -= dt * a[i][j] * function

		return special_function.reshape((N * s))



# multistep methods
# coefs for Adams methods
def build_explicit_adams(order):
	# produces list of order elements
	# coefs go from lower degree to the higher

	s = order

	b_coefs = []
	if s == 1:
		b_coefs = [1]
	else:
		for j in range(s):
			all_roots = [-i for i in range(s) if i != j]

			integrand = poly.Polynomial.fromroots(all_roots)
			integral  = integrand.integ()			
			value     = integral(1.0)		

			minus_one = 1 if j % 2 == 0 else -1
			coef      = minus_one / factorial(j) / factorial(s - j - 1) * value

			b_coefs.append(coef)

		b_coefs.reverse()

	a_coefs = [0 for i in range(len(b_coefs) + 1)]
	a_coefs[-1] = +1
	a_coefs[-2] = -1

	return a_coefs, b_coefs

def build_implicit_adams(order):
	# produces list of orer + 1 elements
	# coefs go from lower degree to the higher

	s = order

	b_coefs = []
	if s == 0:
		b_coefs = [0, 1]
	else:
		for j in range(s + 1):
			all_roots = [-(i - 1) for i in range(s + 1) if i != j]

			integrand = poly.Polynomial.fromroots(all_roots)
			integral  = integrand.integ()			
			value     = integral(1.0)		

			minus_one = 1 if j % 2 == 0 else -1
			coef      = minus_one / factorial(j) / factorial(s - j) * value

			b_coefs.append(coef)

		b_coefs.reverse()

	a_coefs = [0 for i in range(len(b_coefs))]
	a_coefs[-1] = +1
	a_coefs[-2] = -1

	return a_coefs, b_coefs

# coefs for BDF methods
def build_bdf_1():
	a_coefs = [-1, 1]
	b_coefs = [0, 1]

	return np.float64(a_coefs), np.float64(b_coefs)

def build_bdf_2():
	a_coefs = [1/3, -4/3, 1]
	b_coefs = [0, 0, 2/3]

	return np.float64(a_coefs), np.float64(b_coefs)

def build_bdf_3():
	a_coefs = [2/11, 9/11, -18/11, 1]
	b_coefs = [0, 0, 0, 6/11]

	return np.float64(a_coefs), np.float64(b_coefs)

def build_bdf_4():
	a_coefs = [3/25, -16/25, 36/25, -48/25, 1]
	b_coefs = [0, 0, 0, 0, 12/25]

	return np.float64(a_coefs), np.float64(b_coefs)

def build_bdf_5():
	a_coefs = [-12/137, 75/137, -200/137, 300/137, -300/137, 1]
	b_coefs = [0, 0, 0, 0, 0, 60/137]

	return np.float64(a_coefs), np.float64(b_coefs)

def build_bdf_6():
	a_coefs = [10/147, -72/147, 225/147, -400/147, 450/147, -360/147, 1]
	b_coefs = [0, 0, 0, 0, 0, 0, 60/147]

	return a_coefs, b_coefs

def build_bdf(order):
	order = (1 if order < 1 else order)
	order = (6 if order > 6 else order)

	if order == 2:
		return build_bdf_2()
	if order == 3:
		return build_bdf_3()
	if order == 4:
		return build_bdf_4()
	if order == 5:
		return build_bdf_5()
	if order == 6:
		return build_bdf_6()

	return build_bdf_1()


class AdamsExplicitSolver(SolverBase):
	def __init__(self, order, ivp_solver, dt):
		# order >= 1
		# ivp_solver should guarantee appropriate accuracy
		super(AdamsExplicitSolver, self).__init__(ivp_solver.function, ivp_solver.jacobian, ivp_solver.t, ivp_solver.u)
		
		self.ivp_solver = ivp_solver

		self.sys_order = len(ivp_solver.value())

		self.dt = dt

		a, b = build_explicit_adams(order)
		self.coefs = np.float64(b)
		self.steps = len(self.coefs)

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()[:]
			ivp_solver.evolve(self.t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()[:]

		self.t = ivp_solver.t
		self.u[:] = self.values[-1][:]

	def evolve(self, t, dt):
		# dt - ignored

		f  = self.function
		y  = self.values
		a  = self.coefs
		dt = self.dt

		next = y[-1].copy()
		for i in range(self.steps):
			j = (self.steps - 1) - i
			next += dt * a[i] * f(t - j * dt, y[i])

		self.__right_round_like_a_record_baby(next)

		self.t = t + dt # DANGER
		self.u[:] = self.values[-1][:]

	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1][:]
		self.values[-1][:] = value[:]

class AdamsImplicitSolver(SolverBase):
	def __init__(self, order, ivp_solver, neuton_solver, dt):
		# order >= 0
		# ivp_solver should guarantee appropriate accuracy

		super(AdamsImplicitSolver, self).__init__(ivp_solver.function, ivp_solver.jacobian, ivp_solver.t, ivp_solver.u)

		self.ivp_solver = ivp_solver
		self.neuton_solver = neuton_solver

		self.sys_order = len(ivp_solver.value())

		self.dt = dt
		self.A  = None
		self.B  = None

		a, b = build_implicit_adams(order)
		self.coefs = np.float64(b)
		self.steps = len(self.coefs) - 1

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()
			ivp_solver.evolve(self.t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()

		self.t = ivp_solver.t
		self.u[:] = self.values[-1][:]

	def evolve(self, t, dt):
		# dt - ignored

		f = self.function
		y = self.values
		a = self.coefs
		dt = self.dt

		B = y[-1].copy()
		for i in range(self.steps):
			j = (self.steps - 1) - i
			B += dt * a[i] * f(t - j * dt, y[i])
		
		A = dt * a[-1]

		self.t = t
		self.A = A
		self.B = B

		next = y[-1].copy()
		next = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, next)

		self.__right_round_like_a_record_baby(next)

		self.t = t + dt
		self.u = self.values[-1][:]

	def __special_function(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt

		f = self.function

		return y - A * f(t + dt, y) - B
		
	def __special_jacobian(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt

		j = self.jacobian

		special_jacobian = -A * j(t + dt, y)
		for i in range(self.sys_order):			
			special_jacobian[i][i] += 1.0
		return special_jacobian
	
	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1]
		self.values[-1][:] = value

class ExplicitMultistepSolver(SolverBase):
	def __init__(self, ivp_solver, a_coefs, b_coefs, dt):
		# order >= 1
		# ivp_solver should guarantee appropriate accuracy
		super(ExplicitMultistepSolver, self).__init__(ivp_solver.function, ivp_solver.jacobian, ivp_solver.t, ivp_solver.u)
		
		self.ivp_solver = ivp_solver

		self.sys_order = len(ivp_solver.value())

		self.dt = dt

		self.a_coefs = a_coefs.copy()
		self.b_coefs = b_coefs.copy()
		
		self.steps = len(a_coefs) - 1

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()[:]
			ivp_solver.evolve(self.t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()[:]

		self.t = ivp_solver.t
		self.u[:] = self.values[-1][:]

	def evolve(self, t, dt):
		# dt - ignored

		f  = self.function
		y  = self.values
		a  = self.a_coefs
		b  = self.b_coefs
		dt = self.dt

		next = np.zeros((self.sys_order,), np.float64)
		for i in range(self.steps):
			next -= a[i] * y[i]
		for i in range(min(self.steps, len(b))):
			j = (self.steps - 1) - i
			next += dt * b[i] * f(t - j * dt, y[i])
		next /= a[-1]
		self.__right_round_like_a_record_baby(next)

		self.t = t + dt # DANGER
		self.u[:] = self.values[-1][:]

	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1][:]
		self.values[-1][:] = value[:]

class ImplicitMultistepSolver(SolverBase):
	def __init__(self, ivp_solver, neuton_solver, a_coefs, b_coefs, dt):
		# order >= 0
		# ivp_solver should guarantee appropriate accuracy

		super(ImplicitMultistepSolver, self).__init__(ivp_solver.function, ivp_solver.jacobian, ivp_solver.t, ivp_solver.u)

		self.ivp_solver = ivp_solver
		self.neuton_solver = neuton_solver

		self.sys_order = len(ivp_solver.value())

		self.dt = dt
		self.A  = None
		self.B  = None

		self.a_coefs = a_coefs.copy()
		self.b_coefs = b_coefs.copy()
		self.steps = len(self.a_coefs) - 1

		self.values = np.zeros((self.steps, self.sys_order), np.float64)
		for i in range(self.steps - 1):
			self.values[i][:] = ivp_solver.value()
			ivp_solver.evolve(self.t + i * dt, dt)
		self.values[-1][:] = ivp_solver.value()

		self.t = ivp_solver.t
		self.u[:] = self.values[-1][:]

	def evolve(self, t, dt):
		# dt - ignored

		f = self.function
		y = self.values
		a = self.a_coefs
		b = self.b_coefs
		dt = self.dt

		B = np.zeros((self.sys_order,), np.float64)
		for i in range(self.steps):
			B += a[i] * y[i]
		for i in range(min(self.steps, len(b))):
			j = (self.steps - 1) - i
			B -= dt * b[i] * f(t - j * dt, y[i])
		
		A = dt * (b[-1] if len(a) == len(b) else 0)

		self.t = t
		self.A = A
		self.B = B

		next = y[-1].copy()
		next = self.neuton_solver.solve(self.__special_function, self.__special_jacobian, next)

		self.__right_round_like_a_record_baby(next)

		self.t = t + dt
		self.u = self.values[-1][:]

	def __special_function(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt
		a, b  = self.a_coefs, self.b_coefs
		f = self.function

		return a[-1] * y + B - A * f(t + dt, y)
		
	def __special_jacobian(self, y):
		A, B  = self.A, self.B
		t, dt = self.t, self.dt
		a, b  = self.a_coefs, self.b_coefs
		j = self.jacobian

		special_jacobian = -A * j(t + dt, y)
		for i in range(self.sys_order):			
			special_jacobian[i][i] += 1.0 * a[-1]
		return special_jacobian
	
	def __right_round_like_a_record_baby(self, value):
		for i in range(self.steps - 1):
			self.values[i][:] = self.values[i + 1]
		self.values[-1][:] = value


# solver wrapper for automatic integration
# TEST
# do not use with multistep solvers
class AutoSolver(SolverBase):
	def __init__(self, ivp_solver, coef_increase = 2.0, eps = 1e-15, order = 1):
		super(AutoSolver, self).__init__(ivp_solver.function, ivp_solver.jacobian, ivp_solver.t, ivp_solver.u)

		self.ivp_solver = ivp_solver
		self.coef_increase = coef_increase
		self.eps = abs(eps)

	def evolve(self, t, dt):
		eps = self.eps
		inc    = self.coef_increase
		solver = self.ivp_solver

		_, u0 = solver.get_state()
		t0  = t
		dt0 = dt
		tn  = t + dt
		while t0 < tn:
			if t0 + dt0 > tn:
				dt0 = tn - t0

			solver.set_state(t0, u0)
			solver.evolve(t0, dt0)
			t1, u1 = solver.get_state()

			solver.set_state(t0, u0)
			solver.evolve(t0          , dt0 / 2)
			solver.evolve(t0 + dt0 / 2, dt0 / 2)
			t2, u2 = solver.get_state()
		
			curr_eps = abs(u1 - u2).max()
			if curr_eps < eps:
				u0[:] = u2[:]
				t0 = t0 + dt0
				dt0 *= inc
			else:
				dt0 /= 2

		self.t = t + dt
		self.u[:] = u0[:]
				

#tests
def test_equation(t, u):
	v, w = u[0], u[1]

	res = np.zeros((2,), np.float64)
	res[0] = w
	res[1] = m.exp(t) - 2 * w - v

	return res

def test_equation1(t, u):
	res = np.zeros((1,), np.float64)
	res[0] = 2 * t * u
	return res

def solution(t):
	em = m.exp(-t)
	ep = m.exp(+t)

	return np.float64((em * t + ep / 4, em * (-t + 1) + ep / 4))

def solution1(t):
	res = np.zeros((1,), np.float64)
	res[0] = m.exp(t * t) 
	return res


def test_solver(solver, start, end, t0, dt):
	for i in range(start, end):
		solver.evolve(t0 + i * dt, dt)

def test_adams_coefs():
	print(build_explicit_adams(4))
	print(build_implicit_adams(4))

def test_rki_solver():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = DiffJacobian2(f, 1e-7)

	ivp_solver = RKI_naive(lobattoIIIC_4(), f, j, NeutonSolver(1e-15, 100), t0, u0)

	test_solver(ivp_solver, 0, 1, t0, dt)

	print(ivp_solver.t)
	print(ivp_solver.value() - un)

def test_rke_solver():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = None

	ivp_solver = RKE(classic_4(), f, t0, u0)

	test_solver(ivp_solver, 0, n, t0, dt)

	print(ivp_solver.t)
	print(ivp_solver.value() - un)

def test_multistep_solver_explicit():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = DiffJacobian2(f, 1e-7)

	ivp_solver = RKI_naive(lobattoIIIC_4(), f, j, NeutonSolver(1e-15, 100), t0, u0)

	a_coefs, b_coefs = build_explicit_adams(4)
	ems = ExplicitMultistepSolver(ivp_solver, a_coefs, b_coefs, dt)

	test_solver(ems, ems.steps - 1, n, t0, dt)

	print(ems.t)
	print(ems.value() - un) 

def test_multistep_solver_implicit():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = DiffJacobian2(f, 1e-7)

	ivp_solver = RKI_naive(lobattoIIIC_4(), f, j, NeutonSolver(1e-15, 100), t0, u0)

	a_coefs, b_coefs = build_explicit_adams(4)
	ims = ImplicitMultistepSolver(ivp_solver, NeutonSolver(1e-15, 100), a_coefs, b_coefs, dt)

	test_solver(ims, ims.steps - 1, n, t0, dt)

	print(ims.t)
	print(ims.value() - un) 

def test_adams_solver_explicit():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = None

	ivp_solver = RKE(classic_4(), f, t0, u0)

	ems = AdamsExplicitSolver(6, ivp_solver, dt)

	test_solver(ems, ems.steps - 1, n, t0, dt)

	print(ems.t)
	print(ems.value() - un)

def test_adams_solver_implicit():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = DiffJacobian2(f, 1e-7)

	ivp_solver = RKI_naive(lobattoIIIC_4(), f, j, NeutonSolver(1e-15, 100), t0, u0)

	ims = AdamsImplicitSolver(6, ivp_solver, NeutonSolver(1e-15, 100), dt)

	test_solver(ims, ims.steps - 1, n, t0, dt)

	print(ims.t)
	print(ims.value() - un)

# TEST
def test_automatic_solver():
	t0 = 0.0
	u0 = solution(t0)
	tn = 1.0
	un = solution(tn)

	n  = 1000
	dt = (tn - t0) / n

	f = test_equation
	j = DiffJacobian2(f, 1e-7)

	ivp_solver = RKI_naive(lobattoIIIC_4(), f, j, NeutonSolver(1e-15, 100), t0, u0)

	auto_solver = AutoSolver(ivp_solver, 2, 1e-3)

	test_solver(auto_solver, 0, 1, t0, dt)

	print(auto_solver.t)
	print(auto_solver.value() - un) 

def run_tests():
	print('RKI')
	test_rki_solver()
	print('RKE')
	test_rke_solver()
	print('EMS')
	test_multistep_solver_explicit()
	print('IMS')
	test_multistep_solver_implicit()
	print('AES')
	test_adams_solver_explicit()
	print('AIS')
	test_adams_solver_implicit()
	print('AUTO')
	test_automatic_solver()


if __name__ == '__main__':
	run_tests()