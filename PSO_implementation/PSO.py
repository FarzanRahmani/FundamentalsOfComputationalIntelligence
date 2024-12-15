# The function to be optimized
def g(x):
    return -1*(x-5)**2 + 10

def find_best_global(p1, p2):
    val_p1 = g(p1)
    val_p2 = g(p2)

    if val_p1 >= val_p2:
        return p1
    else:
        return p2

# PSO Parameters
omega = 0.5
c1 = 1.5
c2 = 1.5
r1 = 0.4
r2 = 0.7

# Particle 1 initial values
p1_position = 2
p1_velocity = 1
p1_personal_best = p1_position  # Assume personal best = initial position

# Particle 2 initial values
p2_position = 11
p2_velocity = -2
p2_personal_best = p2_position  # Assume personal best = initial position

# Global initial value 
# global_best = 5  # g
global_best = find_best_global(p1_personal_best, p2_personal_best)  # g

print(f"p1_position: {p1_position}")
print(f"p1_velocity: {p1_velocity}")
print(f"p2_position: {p2_position}")
print(f"p2_velocity: {p2_velocity}")
print(f"global_best: {global_best}")
print("-------------------------------------------------")
# ----------------------------------------------------------------------------- #

# Velocity update for Particle 1
p1_velocity_new = (
    omega * p1_velocity
    + c1 * r1 * (p1_personal_best - p1_position)
    + c2 * r2 * (global_best - p1_position)
)
# Position update for Particle 1
p1_position_new = p1_position + p1_velocity_new
if g(p1_position_new) > g(p1_personal_best):
    p1_personal_best = p1_position_new

# Velocity update for Particle 2
p2_velocity_new = (
    omega * p2_velocity
    + c1 * r1 * (p2_personal_best - p2_position)
    + c2 * r2 * (global_best - p2_position)
)
# Position update for Particle 2
p2_position_new = p2_position + p2_velocity_new
if g(p2_position_new) > g(p2_personal_best):
    p2_personal_best = p2_position_new

# update global best
global_best = find_best_global(p1_personal_best, p2_personal_best)  # g

# Print results
print(f"p1_velocity_new: {p1_velocity_new}")
print(f"p1_position_new: {p1_position_new}")
print(f"p1_personal_best: {p1_personal_best}")
print(f"p2_velocity_new: {p2_velocity_new}")
print(f"p2_position_new: {p2_position_new}")
print(f"p2_personal_best: {p2_personal_best}")
print(f"global_best: {global_best}")
print("-------------------------------------------------")


# ----------------------------------------------------------------------------- #
p1_position = p1_position_new
p1_velocity = p1_velocity_new
p2_position = p2_position_new
p2_velocity = p2_velocity_new

# Velocity update for Particle 1
p1_velocity_new = (
    omega * p1_velocity
    + c1 * r1 * (p1_personal_best - p1_position)
    + c2 * r2 * (global_best - p1_position)
)
# Position update for Particle 1
p1_position_new = p1_position + p1_velocity_new
if g(p1_position_new) > g(p1_personal_best):
    p1_personal_best = p1_position_new

# Velocity update for Particle 2
p2_velocity_new = (
    omega * p2_velocity
    + c1 * r1 * (p2_personal_best - p2_position)
    + c2 * r2 * (global_best - p2_position)
)
# Position update for Particle 2
p2_position_new = p2_position + p2_velocity_new
if g(p2_position_new) > g(p2_personal_best):
    p2_personal_best = p2_position_new

# update global best
global_best = find_best_global(p1_personal_best, p2_personal_best)  # g

# Print results
print(f"p1_velocity_new: {p1_velocity_new}")
print(f"p1_position_new: {p1_position_new}")
print(f"p1_personal_best: {p1_personal_best}")
print(f"p2_velocity_new: {p2_velocity_new}")
print(f"p2_position_new: {p2_position_new}")
print(f"p2_personal_best: {p2_personal_best}")
print(f"global_best: {global_best}")
print("-------------------------------------------------")


# ----------------------------------------------------------------------------- #
p1_position = p1_position_new
p1_velocity = p1_velocity_new
p2_position = p2_position_new
p2_velocity = p2_velocity_new

# Velocity update for Particle 1
p1_velocity_new = (
    omega * p1_velocity
    + c1 * r1 * (p1_personal_best - p1_position)
    + c2 * r2 * (global_best - p1_position)
)
# Position update for Particle 1
p1_position_new = p1_position + p1_velocity_new
if g(p1_position_new) > g(p1_personal_best):
    p1_personal_best = p1_position_new

# Velocity update for Particle 2
p2_velocity_new = (
    omega * p2_velocity
    + c1 * r1 * (p2_personal_best - p2_position)
    + c2 * r2 * (global_best - p2_position)
)
# Position update for Particle 2
p2_position_new = p2_position + p2_velocity_new
if g(p2_position_new) > g(p2_personal_best):
    p2_personal_best = p2_position_new

# update global best
global_best = find_best_global(p1_personal_best, p2_personal_best)  # g

# Print results
print(f"p1_velocity_new: {p1_velocity_new}")
print(f"p1_position_new: {p1_position_new}")
print(f"p1_personal_best: {p1_personal_best}")
print(f"p2_velocity_new: {p2_velocity_new}")
print(f"p2_position_new: {p2_position_new}")
print(f"p2_personal_best: {p2_personal_best}")
print(f"global_best: {global_best}")
print("-------------------------------------------------")


# ----------------------------------------------------------------------------- #
