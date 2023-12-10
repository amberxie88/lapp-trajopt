import numpy as np

class Problem:
	def __init__(self, name, start_state, goal_state, num_steps, state_dim,
				min_velocity, max_velocity, min_dist, min_state, max_state):
		self.start_state = np.array(start_state)
		self.goal_state = np.array(goal_state)
		self.num_steps = num_steps
		self.state_dim = state_dim
		self.min_velocity = min_velocity
		self.max_velocity = max_velocity
		self.min_dist = min_dist
		self.min_state = min_state
		self.max_state = max_state
		self.init()

	def process_solution(self, soln):
		formatted_soln = soln.reshape((self.num_steps, self.state_dim))
		return formatted_soln

	def init(self):
		# smoothness constraint
		self.x_dim = self.num_steps * self.state_dim 
		self.init_P()
		self.init_G()
		self.init_A_b()
		self.init_initial_guess()

	def init_P(self):
		"""
		Initialize cost matrix P 
		s.t. f = 1/2 x^T P x = sum ||x_{t+1} - x_t||^2
		"""
		shape = self.x_dim
		self.P = np.zeros((shape, shape))
		i, j = np.indices(self.P.shape)
		# np.fill_diagonal(A, [1] * param_a + [2] * (shape - 2 * param_a) + [1] * param_a)
		np.fill_diagonal(self.P,
		             	[1] * self.state_dim + [2] * (shape - 2 * self.state_dim) + [1] * self.state_dim)

		self.P[i == j - self.state_dim] = -2.0
		self.q = np.zeros(shape)

		# Make symmetric and not indefinite
		self.P = (self.P + self.P.T) + 1e-08 * np.eye(self.P.shape[1])

	def init_G(self):
		"""
		satisfy state and velocity limits by formulating 
		lbG <= G^T x <= ubG
		"""
		velocity_mat, state_mat = self.get_velocity_matrix(), self.get_state_matrix()
		# self.G: [self.state_dim * (2 * self.num_steps - 1), self.state_dim * self.num_steps]
		self.G = np.vstack((velocity_mat, state_mat))

		min_velocity, max_velocity = self.get_velocity_limits()
		min_state, max_state = self.get_state_limits()
		self.lbG = np.hstack((min_velocity, min_state))
		self.ubG = np.hstack((max_velocity, max_state))

	def get_velocity_matrix(self):
		# velocity matrix: [self.state_dim * (self.num_steps - 1), self.state_dim * self.num_steps]
		velocity_matrix = np.zeros((self.x_dim, self.x_dim))
		np.fill_diagonal(velocity_matrix, -1.0)
		i, j = np.indices(velocity_matrix.shape)
		velocity_matrix[i == j - self.state_dim] = 1.0

		# to slice zero last row
		velocity_matrix.resize(velocity_matrix.shape[0] - self.state_dim, velocity_matrix.shape[1])
		return velocity_matrix

	def get_state_matrix(self):
		# state matrix: [self.state_dim * self.num_steps, self.state_dim * self.num_steps]
		state_matrix = np.eye(self.x_dim)
		return state_matrix

	def get_velocity_limits(self):
		min_velocity = np.ones(self.x_dim - self.state_dim) * self.min_velocity
		max_velocity = np.ones(self.x_dim - self.state_dim) * self.max_velocity
		return min_velocity, max_velocity

	def get_state_limits(self):
		min_state = np.ones(self.x_dim) * self.min_state
		max_state = np.ones(self.x_dim) * self.max_state
		return min_state, max_state

	def init_A_b(self):
		# satisfy our start and end specification through formulating Ax = b
		# A is a (2 * state_dim, x_dim) matrix with 1s in the first state_dim and last state_dim columns
		A = np.zeros((2 * self.state_dim, self.x_dim))
		i, j = np.indices(A.shape)

		A[i == j] = [1] * self.state_dim + [0] * self.state_dim
		A[i == j - (self.x_dim - 2* self.state_dim) ] = [0] * self.state_dim + [1] * self.state_dim
		A = np.vstack([A])

		# b is a (2 * state_dim, 1) matrix with the start and end state
		b = np.zeros((2 * self.state_dim, 1))
		b[:self.state_dim] = self.start_state[:, None]
		b[self.state_dim:] = self.goal_state[:, None]

		self.A, self.b = A, b

	def init_initial_guess(self):
		# linear interpolating between start and end state as an initial guess for planning the trajectory
		initial_guess = []
		for i in range(self.state_dim):
			start_state = self.start_state[i]
			end_state = self.goal_state[i]
			initial_guess.append(np.linspace(start_state, end_state, self.num_steps))
		initial_guess = np.array(initial_guess)
		initial_guess = initial_guess.transpose()
		self.initial_guess = initial_guess.flatten()