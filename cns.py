


from net import evaluate

ni = 2
no = 1
nl = 2

genome = np.zeros(ni * ni * nl + ni * no)

i = np.array([[1], [3]])

print(evaluate(genome, i, ni, nl, no))

def nn_error(g):
    err = 0
    for k,v in [(np.asarray([[0], [0]]), 0),
                (np.asarray([[1], [0]]), 1),
                (np.asarray([[0], [1]]), 1),
                (np.asarray([[1], [1]]), 0)]:
        err += abs(v - evaluate(g.genome, k, ni, nl, no)[0][0])
    return err

ga = GeneticAlgorithm(nn_error, ni * ni * nl + ni * no)
ga.train()
