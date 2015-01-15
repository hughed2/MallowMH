import random
import itertools
import numpy

#Given two arrays, x and y, calculate the tau distance
def kendalltau(x,y):
    distance = 0
    for i in itertools.combinations(y,2):
        if (x.index(i[0]) > x.index(i[1])):
            distance = distance + 1
    return distance

#Kendall tau given a table rather than an array
def kendalltau_table(v, D):
    distance = 0
    for i in itertools.combinations(v,2):
        distance = distance + D[i[1]][i[0]]
    return distance    

class metropolis_hastings:

    params = {}

    #Transition function. Randomly chooses a pair of neighbors to swap
    def transition_swap_neighbors(self, x):
        swap = random.randint(0, len(x)-2)
        x_p = list(x)
        x_p[swap + 1], x_p[swap] = x_p[swap], x_p[swap + 1]
        return x_p

    #Determines whether to move to x_p or not
    #Standard implementation would have the distribution zero out when kt distance gets too high
    #So instead of probabilities, we use the kendall-tau distance. So we need to adapt the move function.
    def move_kendalltau(self, kt_x, kt_x_p, ratio):
        if self.params["phi"]:
            phi = self.params["phi"]
        else:
            phi = 0.2
        if(kt_x_p < kt_x): #Always move if 
            return True
        if(random.random()*ratio < pow(phi, kt_x_p - kt_x)):
            return True
        return False

    #The default move function uses actual probabilities
    def move_default(self, prob_x, prob_x_p, ratio):
        if(prob_x_p > prob_x):
            return True
        if(random.random()*ratio < prob_x_p / prob_x):
            return True
        return False

    def get_random_L(self, k):
        x = range(k)
        random.shuffle(x)
        return x

    def get_random_B(self, k):
        x = [[0 for i in range(k)] for j in range(k)] #Create blank matrix
        for i in range(0, k-1):
            for j in range(i+1, k):
                x[i][j] = random.random() > 0.5
        return x
        
    #Default the prior distribution to uniform
    def uniform_prior(self, sampleA, sampleB):
        return 1

    #Set random parameter to L(C) or B(C)
    def set_random(self, function):
        functions = {"L(C)" : self.get_random_L, "B(C)" : self.get_random_B}
        self.get_random = functions[function]

    get_random = get_random_L
    prior = uniform_prior

    #Given a preference table, run the M-H algorithm
    def approximate(self, preferences, k, parameters, ground="", trans_f="", loss_f="", prior_f="",
                                                dist_f="", move_f = "", space="L(C)"):
        if trans_f:
            self.transition = trans_f #Set all of our parameters and functions
        else:
            self.transition = self.transition_swap_neighbors
        if loss_f:
            self.loss = loss_f
        else:
            self.loss = kendalltau
        if dist_f:
            self.distribution = dist_f
        else:
            self.distribution = kendalltau_table
        if move_f:
            self.should_move = move_f
        else:
            self.should_move = self.move_kendalltau
        if prior_f:
            self.prior = prior_f
        self.set_random(space)
        

        self.params = parameters

        
        t = 0

        #Start with a random x0
        x = self.get_random(k)

        prob_x = self.distribution(x, preferences)

        #Perform a random walk
        while t < pow(k-2, 2) * 1.8:
            t = t + 1

            #Calculate x'
            x_p = self.transition(x)
            
            #Calculate the acceptance rate and determine whether to step
            prob_x_p = self.distribution(x_p, preferences)
            ratio = self.prior(x, x_p)
            if(self.should_move(prob_x, prob_x_p, ratio)):
                (x, prob_x) = (x_p, prob_x_p)

        if ground:
            return self.loss(ground, x)
        return x