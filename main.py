import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import timeit


# model constants
g = 9.80665
l = 1
ixx = 1
m_cart = 1

fig, ax = plt.subplots(1,3)
ax[1].set_yscale('log')
fig.show()
def theta_dd(theta, dtheta, x_dd):
    return (g*l*np.sin(theta)+x_dd*l*np.cos(theta))/(ixx+l**2)-dtheta/100


def pendulum_dynamics(individual, plot=False):
    def sign():
        if np.random.random() < 0.5:
            return 1
        else:
            return -1
    # unpack the gains vector
    theta_kp, theta_kd, x_kp, x_kd = individual.genotype
    
    # simulation constants
    theta, dtheta, ddtheta = 0.5, 0, 0                     # initial theta conditions
    x, dx, ddx = 0.0, 0, 0                                 # initial position conditions

    theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, np.random.normal(1,0.0), 0      # target position
    # theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, np.random.uniform(-10,10)*sign(), 0      # target position
    dt, t = 0.02, 0                                        # timestep and time

    x_list, theta_list, dtheta_list = [], [], []
    lens = 1000
    for _ in range(lens):
        # dynamics stuff
        # theta_tgt = max(min((x-x_tgt)**3*t_tgt_k, t_tgt_cap), -t_tgt_cap)
        ddx = (theta-theta_tgt)*theta_kp+(dtheta-dtheta_tgt)*theta_kd+(x-x_tgt)*x_kp+(dx-dx_tgt)*x_kd
        # ddx = -theta*g*2-dtheta*4+(x-x_tgt)*0.6+(dx-dx_tgt)*0.8
        # integration stuff
        dx += ddx*dt
        x += dx*dt

        ddtheta = theta_dd(theta, dtheta, ddx)
        dtheta += ddtheta*dt
        theta += dtheta*dt
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # store states
        x_list.append(x)
        theta_list.append(theta)
        dtheta_list.append(dtheta)

    x_list, theta_list, dtheta_list = np.array(x_list), np.array(theta_list), np.array(dtheta_list)
    extreme_theta = np.where(np.abs(theta_list) < 1, np.abs(theta_list), 999)
    score = 15*np.sum(np.abs(x_list)[len(x_list)//2:-1]) + np.sum(extreme_theta[:-1]) + np.sum(np.abs(dtheta_list)[:-1])
    
    # if at any point the extreme theta is at 10, set flag to True
    flag = False
    if np.any(extreme_theta == 999):
        flag = True

    # updating individual attributes
    individual.unstable = flag
    individual.fitness = score
    
    if plot:
        ax[0].clear()
        ax[0].set_ylim((-4, 4))
        ax[0].plot(theta_list, label='extreme theta')
        ax[0].legend()
        ax[1].clear()
        ax[1].set_ylim((-1, 5))
        ax[1].plot(x_list, label='position')
        ax[1].hlines(x_tgt, 0, lens, label='target position')
        ax[1].legend()

        # plt.figure()
        # plt.plot(x_list, label='x')
        # plt.legend()
        # plt.figure()
        # plt.plot(theta_list, label='theta')
        # plt.legend()
        # plt.figure()
        # plt.plot(extreme_theta, label='extreme theta')
        # plt.legend()
        # plt.show()

    return score


class Individual_1:
    def __init__(self, num_dimensions=1):
        self.num_dimensions      = num_dimensions
        self.genotype            = None
        self.strategy_parameters = None
        self.fitness             = None

    def procreate(self, num_offsprings=1):
        epsilon = 10**(-5)
        offsprings = []
        z = np.random.normal(0,1/self.num_dimensions)
        for _ in range(num_offsprings):
            offspring = Individual_1(self.num_dimensions)
            offspring.genotype = np.array([self.genotype[i]+np.random.normal(0,self.strategy_parameters[i]) for i in range(self.num_dimensions)])
            zi = np.random.normal(0,1/2/np.sqrt(self.num_dimensions))
            offspring.strategy_parameters = np.array([max(sigma*np.exp(z+zi), epsilon) for sigma in self.strategy_parameters])
            offsprings.append(offspring)
        return offsprings
    
class ES_1:
    def __init__(self, fitness_function=None, num_dimensions=1, num_generations=100, num_individuals=50, num_offspring_per_individual=5, verbose=False):
        self.fitness_function = fitness_function
        self.num_dimensions   = num_dimensions
        self.num_generations  = num_generations
        self.num_individuals  = num_individuals
        self.num_offspring_per_individual = num_offspring_per_individual
        self.verbose          = verbose
        self.noconfidence_stretch = 0

        assert fitness_function is not None, "Fitness function needs to be defined"

    def NIS(self):
        # if best individual is unstable, increment noconfidence_stretch
        if self.best.unstable:
            if self.verbose: print(f'    best individual is unstable')
            self.noconfidence_stretch += 1.5
        # if best individual is unstable, increment noconfidence_stretch
        elif abs(self.delta_best_fitness) < 500:
            if self.verbose: print(f'    best individual is not improving')
            self.noconfidence_stretch += 0.8
        # if best individual is stable and improving, reset noconfidence_stretch
        else:
            if self.verbose: print(f'    best individual is stable????')
            self.noconfidence_stretch -= 3
            self.noconfidence_stretch = max(self.noconfidence_stretch, 0)
        
        # if noconfidence_stretch is too large, reset every individual's strategy parameters
        if self.noconfidence_stretch > 10:
            if self.verbose: print(f'    noconfidence_stretch is too large, resetting all strategy parameters\n\n')
            for individual in self.population:
                individual.strategy_parameters = np.maximum(np.random.normal(3,7, size=self.num_dimensions), 0.01)
            self.noconfidence_stretch = 0

    def run(self):
        self.population = [self.generate_random_individual() for _ in range(self.num_individuals)]
        self.best = sorted(self.population, key=lambda individual: self.fitness_function(individual))[0]
        best_fits = []
        avg_fits  = []
        std_fits  = []
        for generation in range(self.num_generations):
            offsprings = []
            for parent in self.population:
                offsprings += parent.procreate(self.num_offspring_per_individual)

            # print(f'size of pop: {len(population)}')
            self.population += offsprings
            # print(f'size of pop after: {len(population)}\n')
            self.population = sorted(self.population, key=lambda individual: self.fitness_function(individual))[:self.num_individuals]

            prev_best = self.best
            self.best = self.population[0]
            self.delta_best_fitness = self.best.fitness - prev_best.fitness

            self.NIS()
            
            best_fits.append(self.best.fitness)
            fitness_list = [individual.fitness for individual in self.population]
            avg_fits.append(np.mean(fitness_list))
            std_fits.append(np.std(fitness_list))

            ax[2].clear()
            ax[2].plot(best_fits, label='best loss')
            ax[2].plot(avg_fits, label='avg loss')
            ax[2].set_yscale('log')
            ax[2].fill_between(np.arange(len(best_fits)), [avg_fits[i]+std_fits[i] for i in range(len(std_fits))], [avg_fits[i]-std_fits[i] for i in range(len(std_fits))], label=r'1 $\sigma$', alpha=0.5)
            ax[2].set_title(f'Generation {generation}')
            ax[2].legend()
            fig.canvas.draw()
            fig.canvas.flush_events()
            if self.verbose:
                if self.verbose: print(f"[gen {generation:3}] Best fitness: {self.fitness_function(self.best, plot=True)}, Delta_fitness: {self.delta_best_fitness:.2f}")


        return self.best
    
    def generate_random_individual(self):
        # --- Initialize the population here ---
        # - For the genotype, sample a standard random normal distribution for each variable separately
        # - For the strategy parameter, sample a standard random normal distribution and then take the maximum of that sample and 0.1 
        #   (to ensure it is not negative and not too small for exploration)
        ind = Individual_1(self.num_dimensions)
        ind.genotype = np.random.uniform(-15,15, size=self.num_dimensions)
        ind.strategy_parameters = np.maximum(np.random.normal(2,5, size=self.num_dimensions), 0.01)
        return ind


class Individual:
    def __init__(self, genotype, strategy_parameters):
        self.genotype = genotype
        self.strategy_parameters = strategy_parameters

class ES:
    def __init__(self, fitness_function=lambda x: 0, num_dimensions=1, 
                 num_generations=100, num_individuals=50, 
                 num_offspring_per_individual=5, verbose=False):
        self.fitness_function = fitness_function
        self.num_dimensions = num_dimensions
        self.num_generations = num_generations
        self.num_individuals = num_individuals
        self.num_offspring_per_individual = num_offspring_per_individual
        self.verbose = verbose
        
        assert num_individuals % 2 == 0, "Population size needs to be divisible by 2 for cross-over"
    
    def run(self):
        population = [self.generate_random_individual() for _ in range(self.num_individuals)]
        best = sorted(population, key=lambda individual: self.fitness_function(individual))[0]
        print(population[1])
        for generation in range(self.num_generations):
            # --- Perform mutation and selection here ---
            # - Each parent individual should produce `num_offspring_per_individual` children by mutation
            #   (recombination is ignored for this exercise)
            # - Implement P+O (parent+offspring) with truncation selection (picking the best n individuals)
            # - Update the `best` variable to hold the best individual of this generation (to then be printed below)
            

            offsprings = []
            for parent in population:
                for _ in range(self.num_offspring_per_individual):
                    parent_genotype = parent.genotype
                    parent_strategy_parameter = parent.strategy_parameters[0]
                    new_genotype = np.array([parent_genotype[i]+np.random.normal(0,parent_strategy_parameter) for i in range(self.num_dimensions)])
                    new_strategy_parameters = np.array([max(parent_strategy_parameter*np.exp(np.random.normal(0,1/self.num_dimensions)),10**(-6))])
                    offsprings.append(Individual(new_genotype, new_strategy_parameters))
            population += offsprings
            population = sorted(population, key=lambda individual: self.fitness_function(individual))[:self.num_individuals]
            best = population[0]

            if self.verbose: print(f"[gen {generation:3}] Best fitness: {self.fitness_function(best)}")

        return self.fitness_function(best)
    
    def generate_random_individual(self):
        # --- Initialize the population here ---
        # - For the genotype, sample a standard random normal distribution for each variable separately
        # - For the strategy parameter, sample a standard random normal distribution and then take the maximum of that sample and 0.1 
        #   (to ensure it is not negative and not too small for exploration)
        return Individual(np.array([np.random.uniform(-15,15) for _ in range(self.num_dimensions)]), np.array([max(np.random.normal(2,5), 0.1)]))


num = 0
q = 1
for i in range(q):
    print(f'\n\n\n\nseed {i}')
    np.random.seed(123+i)

    es = ES_1(fitness_function=pendulum_dynamics, num_dimensions=4, num_generations=150, num_individuals=150, num_offspring_per_individual=6, verbose=True)

    best = es.run()

print(num/q)

# model constants
g = 9.80665
l = 1
ixx = 1
m_cart = 1

# simulation constants
theta, dtheta, ddtheta = 0.5, 0, 0      # initial theta conditions
x, dx, ddx = 0.0, 0, 0                    # initial position conditions

theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, 0, 0                    # target position
timestep, t = 0.02, 0                  # timestep and time


best_gains = best.genotype

# pg viewer intialization
res = [700, 300]
target_fps = 90
origin = (res[0]/2, res[1]/2)
screen = pg.display.set_mode(res)
pg.display.set_caption('pendulum')
pg.init()

# Inititalize main sim clock
clock = pg.time.Clock()

# define colors
white = (255, 255, 255)
red = (255, 0, 50)
light_red = (255, 194, 194)
blue = (0, 50, 255)
light_blue = (194, 194, 255)
green = (50, 255, 0)
light_green = (194, 255, 194)
gold = (255, 223, 0)


# lists to store states
theta_list = []
theta_tgt_list = []
dtheta_list = []
dtheta_tgt_list = []
x_list = []
ddx_list = []
x_tgt_list = []
t_list = []


# simulation consts
theta_kp = best_gains[0]
theta_kd = best_gains[1]
x_kp = best_gains[2]
x_kd = best_gains[3]
t_tgt_k = 0
t_tgt_cap = 0
theta_tgt, dtheta_tgt, x_tgt, dx_tgt = 0, 0, 0, 0                    # target position
timestep = 0.1

print("\n\ngains used:  ", best_gains, "\n\n")

## Pygame Simulation Loop###
frame_time = 0
frame_list = []
running = 1
play = 0
while running:
    toc = timeit.default_timer()
    # dt_frame = timestep
    dt_frame = clock.tick(target_fps) / 1000
    # Clear screen

    screen.fill((0, 0, 0))

    dt = dt_frame*play
    t += dt
    t_list.append(t)

    # dynamics stuff
    # theta_tgt = max(min((x-x_tgt)**3*t_tgt_k, t_tgt_cap), -t_tgt_cap)
    ddx = (theta-theta_tgt)*theta_kp+(dtheta)*theta_kd+(x-x_tgt)*x_kp+(dx-dx_tgt)*x_kd
    # ddx = -theta*g*2-dtheta*4+(x-x_tgt)*0.6+(dx-dx_tgt)*0.8
    ddx_list.append(ddx)
    # integration stuff
    dx += ddx*dt
    x += dx*dt

    ddtheta = theta_dd(theta, dtheta, ddx)
    dtheta += ddtheta*dt
    theta += dtheta*dt
    theta = (theta + np.pi) % (2 * np.pi) - np.pi


    x_list.append(x)
    x_tgt_list.append(x_tgt)
    theta_list.append(theta)
    theta_tgt_list.append(theta_tgt)

    # real coords
    o_x = x
    o_y = 0
    tip_x = -np.sin(theta) + x
    tip_y = np.cos(theta)

    # converting real coords into window coords
    win_o_x = res[0]/2 + 100*o_x
    win_o_y = res[1]/2 - 100*o_y
    win_tip_x = res[0]/2 + 100*tip_x
    win_tip_y = res[1]/2 - 100*tip_y
    win_tgt_x = res[0]/2 + 100*x_tgt
    win_tgt_y = res[1]/2

    pg.draw.aaline(screen, white, (win_o_x, win_o_y), (win_tip_x, win_tip_y))
    pg.draw.circle(screen, red, (win_tgt_x, win_tgt_y), 2)
    for i in range(10):
        win_tick_x = res[0]/2 + 100*i
        win_tick_y = res[1]/2
        pg.draw.circle(screen, white, (win_tick_x, win_tick_y), 0.5)

    for event in pg.event.get():
        # Stay in main loop until pygame.quit event is sent
        if event.type == pg.QUIT:
            running = 0
        # If a key on the keyboard is pressed
        elif event.type == pg.KEYDOWN:
            # Escape key, end game
            if event.key == pg.K_ESCAPE:
                running = 0
            if event.key == pg.K_RIGHT:
                x_tgt += 1.5
            if event.key == pg.K_LEFT:
                x_tgt -= 1.5
            if event.key == pg.K_UP:
                theta += 0.2
            if event.key == pg.K_DOWN:
                theta -= 0.2
            if event.key == pg.K_SPACE:
                if play == 0:
                    play = 1
                else:
                    play = 0
    pg.display.flip()

    tic = timeit.default_timer()


    frame_list.append(1000*(tic-toc))
    if len(frame_list) > 200:
        print(f'frame time: {round(sum(frame_list)/len(frame_list), 8)} ms')
        frame_list = []


pg.quit()


fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle('Inverted pendulum :0000')

axs[0,0].plot(t_list, x_list, label='x')
axs[0,0].plot(t_list, x_tgt_list, label='x tgt')
axs[0,0].legend(loc='upper right')
axs[0,0].grid()
axs[0,0].set_ylabel('[m]')
axs[1,0].plot(t_list, theta_list, label='theta')
axs[1,0].plot(t_list, theta_tgt_list, label='theta tgt')
axs[1,0].legend(loc='upper right')
axs[1,0].grid()
axs[1,0].set_ylabel('[rad]')
axs[0,1].plot(t_list, ddx_list, label='ddx')
axs[0,1].legend(loc='upper right')
axs[0,1].grid()
plt.show()


plt.show()