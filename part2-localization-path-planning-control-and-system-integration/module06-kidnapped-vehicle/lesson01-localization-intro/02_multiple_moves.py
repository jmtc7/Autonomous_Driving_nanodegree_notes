#write code that moves 1000 times and then prints the 
#resulting probability distribution.

p=[0, 1, 0, 0, 0]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1

def sense(p, Z):
	"""
	Reduces the undertainty of a given probability distribution using a measurement.

	::param p:: Probability distribution.
	::param Z:: Measurement.
	"""
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
	"""
	Applies motion uncertainty (pExact, pOvershoot and pUndershoot) to a given probability distribution when we move a given amount of cells.

	::param p:: Beliefs in each posible robot position.
	::param U:: Amount of cells that the robot is told to move.
	"""
    q = []
    for i in range(len(p)):
        s = pExact * p[(i-U) % len(p)]
        s = s + pOvershoot * p[(i-U-1) % len(p)]
        s = s + pUndershoot * p[(i-U+1) % len(p)]
        q.append(s)
    return q

# Move 1 cell a 1000 times
n_moves = 1000
n_steps_per_move = 1

for i in range(n_moves):
    p = move(p, n_steps_per_move)

# We get a uniform distribution
print p
