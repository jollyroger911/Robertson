import numpy as np
import heapq as h


# permutation class
class Permutation:
	def __init__(self, size):
		self.size  = size
		self.pariy = 1 
		self.permutation = np.zeros((size,), np.int)
		self.reset()

	def permute(self, i, j):
		if i != j:
			p = self.permutation
			p[i], p[j] = p[j], p[i]
		
			self.parity *= -1

	def reset(self):
		for i in range(self.size):
			self.permutation[i] = i
		self.parity = 1

	def get_parity(self):

		return self.parity

	def get_size(self):

		return self.size

	def __getitem__(self, i):

		return self.permutation[i]


# some utilities
def __find_abs_max(mat, k):
	n = len(mat)

	elem_max = abs(mat[k][k])
	i_max = k
	j_max = k

	for i in range(k, n):
		for j in range(k, n):
			if elem_max < abs(mat[i, j]):
				elem_max = abs(mat[i, j])
				i_max = i
				j_max = j

	return i_max, j_max

def __swap_comps(vec, i, j):

	vec[i], vec[j] = vec[j], vec[i]

def __swap_rows(mat, i, j):
	n = len(mat)
	
	for k in range(n):
		mat[i, k], mat[j, k] = mat[j, k], mat[i, k]

def __swap_cols(mat, i, j):
	n = len(mat)
	
	for k in range(n):
		mat[k, i], mat[k, j] = mat[k, j], mat[k, i]


# gaussian elimination algorithm
def __gaussian_step(A, b, step):
	n = len(A)

	div = A[step, step]

	for i in range(step + 1, n):
		mult = A[i, step]

		for j in range(step, n):
			A[i, j] -= A[step, j] * (mult / div)

		b[i] -= b[step] * (mult / div)

def __forward_step(A, b, cols_perm, rows_perm):
	n = len(A)

	for i in range(n):
		i_max, j_max = __find_abs_max(A, i)

		if abs(A[i_max, j_max]) < 1e-15:
			print('ERROR: ZERO DETERMINANT MATRIX')

		if i_max != i:
			__swap_rows(A, i, i_max)
			__swap_comps(b, i, i_max)

			rows_perm.permute(i, i_max)

		if j_max != i:
			__swap_cols(A, i, j_max)

			cols_perm.permute(i, j_max)

		__gaussian_step(A, b, i)

def __backward_step(A, b, cols_perm, rows_perm):
	n = len(A)

	for i in range(n - 1, -1, -1):
		b[i] /= A[i, i]

		for j in range(i + 1, n):
			b[i] -= b[j] / A[i, i] * A[i, j]

def __restore_component_order(b, cols_perm):
	n = len(b)

	i = 0
	while i < n:
		j = cols_perm[i]

		if i != j:
			__swap_comps(b, i, j)
			cols_perm.permute(i, j)
		else:
			i += 1


def gaussian_elimination_solve(A, b):
	# Ax = b - linear system
	n = len(A)
	A = A.copy()
	b = b.copy()

	cols_perm = Permutation(n)
	rows_perm = Permutation(n)

	__forward_step(A, b, cols_perm, rows_perm)
	__backward_step(A, b, cols_perm, rows_perm)
	__restore_component_order(b, cols_perm)

	return b



def main():
	A = np.zeros((5,5), np.float64)
	b = np.zeros((5,), np.float64)

	A[0, 0] = -1e10; A[0, 1] = 9e20;    A[0, 2] = 0; A[0, 3] = 0; A[0, 4] = 0;
	A[1, 0] = +1e10; A[1, 1] = 9e20;    A[1, 2] = 0; A[1, 3] = 0; A[1, 4] = 0;
	A[2, 0] = 1; A[2, 1] = 0;    A[2, 2] = 1; A[2, 3] = 0; A[2, 4] = 0;
	A[3, 0] = 1; A[3, 1] = 0;    A[3, 2] = 0; A[3, 3] = 1; A[3, 4] = 0;
	A[4, 0] = 1; A[4, 1] = 0;    A[4, 2] = 0; A[4, 3] = 0; A[4, 4] = 1;

	b = A @ np.ones((5,), np.float64)

	b1 = gaussian_elimination_solve(A, b)
	b2 = np.linalg.solve(A, b)
	
	print(b1 - b2)


	A = np.zeros((3, 3), np.float64)

	A[0,0] = +1e42; A[0,1] = 1; A[0,2] = 1e30;
	A[1,0] = 0;     A[1,1] = 1; A[1,2] = 1e30;
	A[2,0] = -1e42; A[2,1] = 0; A[2,2] = 1;

	b = A @ np.ones((3,), np.float64)

	b1 = gaussian_elimination_solve(A, b)
	b2 = np.linalg.solve(A, b)
	
	print(b1 - b2)

	


if __name__ == '__main__':
	main()