# Gradient error checking -- possibly incorrect
# all_diff = np.array([])
# for i in range(a_out.shape[0]):
#     a_i = a_out[i, :]
#     y_i = y[i, :]
#     diff = sp.approx_fprime(a_i, compute_L, np.finfo(float).eps, y_i)
#     all_diff = np.append(all_diff, diff)
#
# err = np.sqrt(np.sum((dL_dz.reshape((dL_dz.size)) - all_diff) ** 2))

# Gradient checking using Scipy -- does not work
# err = sp.check_grad(compute_L, lambda x, y: d, a_out[0,:], y[0,:])
# print(err)