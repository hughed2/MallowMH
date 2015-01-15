import random
import itertools
import numpy
from MetroHastings import metropolis_hastings

n = 1000            #The number of voters

#This function credit to Shashank on Stackoverflow
#http://stackoverflow.com/questions/19372991/number-of-n-element-permutations-with-exactly-k-inversions
def mahonian_row(n):
    '''Generates coefficients in expansion of 
    Product_{i=0..n-1} (1+x+...+x^i)
    **Requires that n is a positive integer'''
    # Allocate space for resulting list of coefficients?
    # Initialize them all to zero?
    #max_zero_holder = [0] * int(1 + (n * 0.5) * (n - 1))

    # Current max power of x i.e. x^0, x^0 + x^1, x^0 + x^1 + x^2, etc.
    # i + 1 is current row number we are computing
    i = 1
    # Preallocate result
    # Initialize to answer for n = 1
    result = [1]
    while i < n:
        # Copy previous row of n into prev
        prev = result[:]
        # Get space to hold (i+1)st row
        result = [0] * int(1 + ((i + 1) * 0.5) * (i))
        # Initialize multiplier for this row
        m = [1] * (i + 1)
        # Multiply
        for j in range(len(m)):
            for k in range(len(prev)):
                result[k+j] += m[j] * prev[k]
        # Result now equals mahonian_row(i+1)
        # Possibly should be memoized?
        i = i + 1
    return result

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

#Given a ground truth, calculate prefs of n voters with Mallow's model
def calc_prefs(ground, n, phi):

    pref_table = [[0 for i in range(len(ground))] for j in range(len(ground))]
    KT_probs = []


    #Add a voter's preferences to the table
    def add_voter(voter):
        for i in itertools.combinations(voter,2):
            pref_table[i[0]][i[1]] += 1

    #Given a KT distance, create a voter
    def get_voter(dist):
        i = 0
        voter = list(ground)
        while i < dist: #Try to increase KT by 1 per loop
            swap = random.randint(0, k-2)
            if voter[swap] < voter[swap+1]: #Make sure we're actually increasing KT
                voter[swap], voter[swap+1] = voter[swap+1], voter[swap]
                i += 1
        return voter

    #Create probability list for separate KT distances
    def calc_KT_probs():
        KT_probs = mahonian_row(len(ground)) #The number of permutations with a given distance
        for i in range(1, len(KT_probs)):         #Multiply them by the probability of an individual permutation and keep running track
            KT_probs[i] = (pow(phi, i) * float(KT_probs[i])) + KT_probs[i-1]
        for i in range(len(KT_probs)):         #Normalize
            KT_probs[i] = KT_probs[i] / KT_probs[-1]
        return KT_probs

    KT_probs = calc_KT_probs()

    for j in range(n):
        voter = random.random() #Get the kt distance for one voter
        i = 0
        while voter > KT_probs[i]:
            i += 1
        voter = get_voter(i)
        add_voter(voter)
            
    return pref_table

if __name__ == "__main__":

    MH = metropolis_hastings()

    for k in range(3, 26):
        for phi in [0.1,0.2,0.3,0.4,0.5]:
            k_dist = []
            ground = range(k)
            for iter in range(100):
                #We need to calaculate voters first
                pref_table = calc_prefs(ground, n, phi)

                #Now we can run Metropolis-Hastings
                params = {"phi" : phi}
                k_dist.append(MH.approximate(pref_table, k, params, ground))

            #k, phi, mean KT distance, std KT distance, and absolute number resulting in ground truth    
            print k, phi, sum(k_dist)/100.0, numpy.std(numpy.array(k_dist)), k_dist.count(0)
