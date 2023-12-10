import numpy as np
import cvxpy
import copy
import time

class TrajOpt:
	def __init__(self, name, mu_0, s_0, c, tau_plus, tau_minus, k, ftol, xtol, ctol, 
				solver, penalty_max_iteration, convexify_max_iteration, trust_max_iteration,
				min_trust_box_size, use_collision, d_check):
		self.mu_0 = mu_0
		self.s_0 = s_0
		self.c = c
		self.tau_plus = tau_plus
		self.tau_minus = tau_minus
		self.k = k
		self.ftol = ftol
		self.xtol = xtol
		self.ctol = ctol
		self.solver = solver
		self.penalty_max_iteration = penalty_max_iteration
		self.convexify_max_iteration = convexify_max_iteration
		self.trust_max_iteration = trust_max_iteration

		self.s = s_0

		# additional starter values
		self.trust_region_norm = np.inf
		self.min_trust_box_size = min_trust_box_size
		self.use_collision = use_collision
		self.d_check = d_check

	"""
	Initialize Problem
	"""
	def init(self, env, logger, problem):
		self.D = None
		self.lbD = None
		self.ubD = None

		# f = 1/2 x^T P x + q^T x
		self.P, self.q = problem.P, problem.q
		# lbG <= G^T x <= ubG
		self.G, self.lbG, self.ubG = problem.G, problem.lbG, problem.ubG
		# A^T x = b
		self.A, self.b = problem.A, problem.b
		# initial guess
		self.initial_guess = problem.initial_guess
		self.min_dist = problem.min_dist

		# problem
		self.converged = False
		self.step = 0
		self.logger = logger
		self.env = env
		self.problem = problem

	def get_starter_metrics(self):
		metrics = dict(reduction_actual=0, reduction_predicted=0, trust_ratio=0,
						objective_model_at_p_k=0, objective_actual_at_x_k=0, prob_value=0,
						trust_region_size=0, iteration_penalty=0, iteration_convexify=0,
						iteration_trust_region=0, STATUS_objective_function_converged=0,
						STATUS_x_converged=0, STATUS_constraints_satisfied=0, solve_time=0, 
						STATUS_trust_expand=0, STATUS_convex_success=0, cons1=0,
						cons2=0, cons3=0, cons4=0)

	def process_metrics(self, out, actual_reduction, predicted_reduction, trust_ratio, i, j, k):
		metrics = dict()
		if self.step == 0:
			metrics = self.get_starter_metrics() 
		if out == None:
			# assume the rest are None
			return metrics
		else:
			metrics.update(dict(cons1=out['cons1'], cons2=out['cons2'], cons3=out['cons3']))
			if 'cons4' in out.keys():
				metrics['cons4'] = out['cons4']
				# breakpoint()
				# metrics['cons4_model'] 
			metrics.update(dict(reduction_actual=actual_reduction, reduction_predicted=predicted_reduction, trust_ratio=trust_ratio))
			metrics['objective_model_at_p_k'] = out['model_objective_at_p_k'].value
			metrics['objective_actual_at_x_k'] = out['actual_objective_at_x_k'].value
			metrics['convex_solve_time'] = out['solve_time']
			metrics['prob_value'] = out['prob_value'] 
			metrics['trust_region_size'] = self.s
			metrics['iteration_penalty'] = i
			metrics['iteration_convexify'] = j 
			metrics['iteration_trust_region'] = k
		return metrics
		
	"""
	TrajOpt Algorithm 
	"""
	def solve(self):
		# initialize necessary cvxpy parameters
		penalty = cvxpy.Parameter(nonneg=True)
		penalty.value = self.mu_0
		x_0 = self.initial_guess
		x_k = copy.copy(x_0)

		# perturbation p
		p = cvxpy.Variable(self.P.shape[0])
		# p_0: no perturbation
		p_0 = cvxpy.Variable(self.P.shape[0])
		p_0.value = np.zeros(p.shape[0])
		# p_k, last_p_k: perturbations
		p_k = [0] * len(x_0)
		last_p_k = [0] * len(x_0)
		p.value = copy.deepcopy(p_0.value)

		# slack variable
		t = cvxpy.Variable(self.P.shape[0]//2)
		last_t = cvxpy.Variable(t.shape[0])
		last_t.value = copy.deepcopy(t.value)

		print("init", self.is_constraints_satisfied_indiv(x_k, p, self.ctol))

		# PenaltyIteration
		for i in range(self.penalty_max_iteration):
			# ConvexifyIteration
			for j in range(self.convexify_max_iteration):
				# TrustRegionIteration
				for k in range(self.trust_max_iteration):
					if self.use_collision:
						collision_info = self.env.get_collision_info(x_k, self.d_check, self.problem.lang,) 
						self.process_collision_info(collision_info)
					else:
						collision_info = None

					out = self.solve_problem(x_k, penalty, p, last_t, t, self.s, collision_info)
					self.step += 1

					if out['p_k'] is None or t.value is None:
						# take gradient step in previous perturbation direction
						x_k -= last_p_k 
						p.value = last_p_k 
						p_k = copy.deepcopy(last_p_k)
						metrics = self.process_metrics(None, None, None, None, None, None, None)
						metrics['STATUS_convex_success'] = 0
						self.logger.log_metrics(metrics, self.step, ty='trajopt')
					else:
						p_k = out['p_k']
						actual_objective_at_x_plus_p_k = self.get_actual_objective(x_k + p_k, penalty, t, collision_info) 
						model_objective_at_p_0, _, _ = self.convexify_problem(x_k, p_0, last_t, last_t, penalty, collision_info) 

						actual_reduction = out['actual_objective_at_x_k'].value - actual_objective_at_x_plus_p_k.value
						predicted_reduction = model_objective_at_p_0.value - out['model_objective_at_p_k'].value

						if predicted_reduction == 0:
						    predicted_reduction = 0.0000001

						# TrueImprove / ModelImprove
						trust_ratio = actual_reduction / predicted_reduction

						metrics = self.process_metrics(out, actual_reduction, predicted_reduction, trust_ratio, i, j, k)
						metrics['STATUS_convex_success'] = 1
						
						check_constraints, break_this_loop = False, False
						if trust_ratio >= self.c:
							self.s = self.tau_plus * self.s 
							x_k += p_k
							metrics['STATUS_trust_expand'] = 1
							break_this_loop = True 
						else:
							self.s = self.tau_minus * self.s
							metrics['STATUS_trust_expand'] = -1
						if self.s < self.xtol:
							# line 11 -> line 15 in Algorithm 1
							check_constraints = True 
							break_this_loop = True 

						last_p_k = p_k 
						last_t.value = copy.deepcopy(t.value)

						self.logger.log_metrics(metrics, self.step, ty='trajopt')
						if break_this_loop:
							break
						

					
				self.s = np.fmax(self.s, self.min_trust_box_size / (self.tau_minus * 0.5))

				if self.is_objective_function_converged(actual_reduction, self.ftol):
					# if converged according to ftol: actual reduction very small 
					self.logger.log_metrics(dict(STATUS_objective_function_converged=1), self.step, ty='trajopt')
					check_constraints = True
				else:
					self.logger.log_metrics(dict(STATUS_objective_function_converged=0), self.step, ty='trajopt')

				if self.is_x_converged(x_k, p_k, self.xtol):
					# if converged according to xtol 
					self.logger.log_metrics(dict(STATUS_x_converged=1), self.step, ty='trajopt')
					check_constraints = True
				else:
					self.logger.log_metrics(dict(STATUS_x_converged=0), self.step, ty='trajopt')

				if check_constraints:
					break 

			if self.is_constraints_satisfied(x_k, p, self.ctol):
				self.logger.log_metrics(dict(STATUS_constraints_satisfied=1), self.step, ty='trajopt')
				self.converged = True
			else:
				penalty.value *= self.k
				self.logger.log_metrics(dict(STATUS_constraints_satisfied=0), self.step, ty='trajopt')

			if self.converged:
				break

		print("finished iterating. converged:", self.converged)
		print(self.is_constraints_satisfied_indiv(x_k, p, self.ctol))
		if self.use_collision:
			collision_info = self.env.get_collision_info(x_k, self.d_check, self.problem.lang, get_gt=True) 
			self.process_collision_info(collision_info)
			print("distances", collision_info['distances'])
		return dict(soln=x_k, collision_info=collision_info)


	def solve_problem(self, x_k, penalty, p, last_t, t, trust_box_size, collision_info):
		model_objective, actual_objective, cons_meta = self.convexify_problem(x_k, p, last_t, t, penalty, collision_info)
		if collision_info is None:
			constraints = [cvxpy.norm(p, self.trust_region_norm) <= trust_box_size]
		else:
			constraints = [cvxpy.norm(p, self.trust_region_norm) <= trust_box_size, 0 <= t, cons_meta['cons4_model'] <= t]

		problem = cvxpy.Problem(cvxpy.Minimize(model_objective), constraints)
		if self.solver == "CVXOPT":
			start = time.time()
			result = problem.solve(solver=self.solver, warm_start=True, kktsolver=cvxpy.ROBUST_KKTSOLVER, verbose=False)
			end = time.time()
		else:
			start = time.time()
			result = problem.solve(solver=self.solver, warm_start=True, verbose=False, max_iters=100)
			end = time.time()

		out = dict(p_k=p.value, model_objective_at_p_k=model_objective, actual_objective_at_x_k=actual_objective, 
					solver_status=problem.status, prob_value=problem.value, solve_time=end-start)
		out.update(cons_meta)
		return out

	def convexify_problem(self, x_k, p, last_t, t, penalty, collision_info):
		# linear approximation of objective around x_k with perturbation p_k
		cons1_at_xk, cons2_at_xk, cons3_at_xk, cons4_at_xk = self.evaluate_constraints(x_k, collision_info)
		cons1_grad_at_xk, cons2_grad_at_xk, cons3_grad_at_xk, cons4_grad_at_xk = self.get_constraints_gradients(x_k, collision_info)
		cons1_model = cons1_at_xk + cons1_grad_at_xk @ p
		cons2_model = cons2_at_xk + cons2_grad_at_xk @ p
		cons3_model = cons3_at_xk + cons3_grad_at_xk @ p

		cons4_model = 0
		if not collision_info is None:
			cons4_model = cons4_at_xk + cons4_grad_at_xk @ p
			
		# minimize quadratic approximation of objective around x_k with perturbation p_k
		# = f(x_k) + \grad{f(x_k)}^T p + 1/2 p^T \Hessian{f(x_k)} p
		objective_grad_at_xk, objective_hess_at_xk = self.get_objective_gradient_and_hessian(x_k)
		objective_at_xk = self.get_actual_objective(x_k, penalty, last_t, collision_info)
		# for debugging
		model = objective_at_xk.value + objective_grad_at_xk @ p + 0.5 * cvxpy.quad_form(p, objective_hess_at_xk)

		# add \hat{g} and \hat{h} to optimization problem
		model += penalty * (cvxpy.norm(cons1_model, 1) + cvxpy.norm(cons2_model, 1)
		                    + cvxpy.norm(cons3_model, 1))
		if not collision_info is None:
			model += penalty * cvxpy.norm(t, 1)


		cons_meta = dict(cons1=np.linalg.norm(cons1_at_xk), cons2=np.linalg.norm(cons2_at_xk),
		                   cons3=np.linalg.norm(cons3_at_xk))
		if not t.value is None:
			cons_meta['cons4'] = np.linalg.norm(t.value)
		cons_meta['cons4_model'] = cons4_model
		return model, objective_at_xk, cons_meta

	# evaluating constraints for a given solver state
	def evaluate_constraints(self, x_k, collision_info):
		# constraint 1 (joint limits): x <= ubG 
		cons1 = np.subtract(np.matmul(self.G, x_k), self.ubG)
		# constraint 2 (joint limits): x >= lbG
		cons2 = np.add(np.matmul(-self.G, x_k), self.lbG)
		# constraint 3 (goal reaching): Ax = b
		cons3 = np.subtract(np.matmul(self.A, x_k), self.b.flatten())
		cons4 = 0
		if collision_info is not None:
			collision_info = self.env.get_collision_info(x_k, self.d_check, self.problem.lang,) 
			self.process_collision_info(collision_info)
			cons4 = self.min_dist - collision_info['distances']

		return cons1.flatten(), cons2.flatten(), cons3.flatten(), cons4

	# gradient of solver constraint matrices
	def get_constraints_gradients(self, x_k, collision_info):
		cons1_grad = self.G
		cons2_grad = -self.G
		cons3_grad = self.A
		cons4_grad = 0
		if collision_info is not None:
			collision_info = self.env.get_collision_info(x_k, self.d_check, self.problem.lang,) 
			self.process_collision_info(collision_info)
			cons4 = self.min_dist - collision_info['distances']
			cons4_grad = -collision_info['gradients']

		return cons1_grad, cons2_grad, cons3_grad, cons4_grad

	# gradient and hessian of the solver objective function f = x^T G x
	def get_objective_gradient_and_hessian(self, x_k):
		model_grad = 0.5 * np.matmul((self.P + self.P.T), x_k)
		model_hess = 0.5 * (self.P + self.P.T)
		return model_grad, model_hess

	# to get the value of the original objective cost at xk
	def get_actual_objective(self, xk, penalty, t, collision_info):
		x = cvxpy.Variable(self.P.shape[0])
		x.value = copy.copy(xk)
		objective = 0.5 * cvxpy.quad_form(x, self.P) + self.q @ x
		constraints1 = cvxpy.norm(self.G @ x - self.ubG.flatten(), 1)
		constraints2 = cvxpy.norm(-self.G @ x + self.lbG.flatten(), 1)
		constraints3 = cvxpy.norm(self.A @ x - self.b.flatten(), 1)
		constraints4 = 0

		t_temp = cvxpy.Variable(t.shape[0])
		t_temp.value = copy.deepcopy(t.value)
		constraints4 = cvxpy.norm(t_temp, 1)

		objective += penalty * (constraints1 + constraints2 + constraints3 + constraints4)
		return objective

	"""
	Checking Constraints / Convergence
	"""
	# method to check if the given state variable respects the given constraints
	def is_constraints_satisfied(self, x_k, p, tolerance=1e-3):
		constraints_indiv = self.is_constraints_satisfied_indiv(x_k, p, tolerance=tolerance)
		return all(constraints_indiv)

	def is_constraints_satisfied_indiv(self, x_k, p, tolerance=1e-3):
		cons1_cond = np.isclose(np.matmul(self.G, x_k) <= self.ubG, 1, rtol=tolerance, atol=tolerance)
		cons2_cond = np.isclose(np.matmul(self.G, x_k) >= self.lbG, 1, rtol=tolerance, atol=tolerance)
		cons3_cond = np.isclose(np.matmul(self.A, x_k), self.b.flatten(), rtol=tolerance, atol=tolerance)
		if self.use_collision:
			cons4_cond = True
			collision_info = self.env.get_collision_info(x_k, self.d_check, self.problem.lang, get_gt=True)
			cons4_cond = np.all(collision_info['raw_distances'] >= self.min_dist)
		else:
			cons4_cond = True

		return cons1_cond.all(), cons2_cond.all(), cons3_cond.all(), cons4_cond

	# method to check if the objective function has converged
	def is_objective_function_converged(self, objective, tolerance=1e-3):
		return abs(objective) <= tolerance

	# method to check if the solver state variable has converged
	def is_x_converged(self, x_k, p_k, tolerance=1e-3):
		return abs((np.linalg.norm(x_k - (x_k + p_k), np.inf))) <= tolerance

	"""
	Misc
	"""
	def process_collision_info(self, collision_info):
		raw_gradients = collision_info['raw_gradients']
		gs = []
		for i in range(raw_gradients.shape[1]):
			g = np.identity(raw_gradients.shape[0])
			g[np.diag_indices(raw_gradients.shape[0])] = raw_gradients[:, i]
			gs.append(g)
		# g2 = np.identity(raw_gradients.shape[0])
		# g2[np.diag_indices(raw_gradients.shape[0])] = raw_gradients[:, 1]

		interleaved = np.concatenate(gs, axis=1)
		interleaved = interleaved.reshape((10, 10*raw_gradients.shape[1]))
		interleaved[:, ::7] = gs[0]
		interleaved[:, 1::7] = gs[1]
		interleaved[:, 2::7] = gs[2]
		interleaved[:, 3::7] = gs[3]
		interleaved[:, 4::7] = gs[4]
		interleaved[:, 5::7] = gs[5]
		interleaved[:, 6::7] = gs[6]
		collision_info['gradients'] = interleaved 
		collision_info['distances'] = collision_info['raw_distances']
		return
