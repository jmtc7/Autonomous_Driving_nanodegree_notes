#Modify the code so that it updates the probability twice
#and gives the posterior distribution after both 
#measurements are incorporated. Make sure that your code 
#allows for any sequence of measurement of any length.

p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2

def sense(p, Z):
	"""
	Returns a normalized probability distribution given a prior (p) and an observation (Z).
	"""
    q=[]

	# Buil non-normalized probability distribution
    for i in range(len(p)):
		# Check if observation corresponds to environment (hit) or not (miss)
        hit = (Z == world[i])

		# Update probability and append to result (q)
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))

	# Normalize probability distribution
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q
    
# Forward different measurements to the function
for measurement in measurements:
    p = sense(p, measurement)

print p
