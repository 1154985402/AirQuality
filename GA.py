import codecs
import csv
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data
from deap import base, creator, tools
from scipy.stats import bernoulli
import RBFnet.RBFNet as RBFnet
import data.DataLoader
import RBFnet.RBF as rbf
import util

basis_func = rbf.gaussian
random.seed(64)

data_path = 'data/PRSA_Data_20130301-20170228/PRSA_Data_Dongsi_20130301-20170228.csv'
dataset = data.DataLoader.Datalder(data_path)
datas = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=True, drop_last=False)
for i, data in enumerate(datas):
    pass

Xdata = data[0]
ydata = data[1]

def evaluate(individual):
    idx = [i for i, x in enumerate(individual) if x == 1]
    X = Xdata[:, idx]
    rbf = RBFnet.Network(len(idx), 25, basis_func)
    y_hat = rbf.forward(X).squeeze(-1)
    y = ydata
    res = F.mse_loss(y_hat, y)
    return res,


def get_best():
    GEN_LENGTH = 15
    creator.create('FitnessMin', base.Fitness, weights=(-1.0,))  # 优化目标：单变量，求最小值
    creator.create('Individual', list, fitness=creator.FitnessMin)  # 创建Individual类，继承list

    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GEN_LENGTH)
    ind1 = toolbox.Individual()
    print(ind1)

    N_POP = 160
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    toolbox.Population(n=N_POP)
    pop = toolbox.Population(n=N_POP)

    toolbox.register('evaluate', evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize= 3)

    G_N = 100
    CXPB = 0.8
    best = []
    Gs = []
    MUTPB = random.gauss(0, 1)
    for g in range(G_N):
        print("iteration : %d ---------------------------------" % (g+1) )
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    toolbox.mutate(mutant, 0, 1)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            best.append(min(fits).detach().numpy())
            Gs.append(g)

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    util.plot_data_fit(Gs,best)
    return best_ind


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("保存文件成功，处理结束")

