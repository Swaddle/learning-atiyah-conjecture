from torch import randn, empty, stack, dot, argmax, zeros_like
from learning_atiyah import PolyM 
from itertools import reduce 

def one_hot(tensor):
    max_index = argmax(tensor.abs())
    one_hot = zeros_like(tensor)
    one_hot[max_index] = 1
    return one_hot

def gen_random_sample_2d(n_points: int):
    p = randn(n_points,2)
    v = randn(n_points)
    dots = empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) -  p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:,:,0]

    Xi = stack(( (M + xs).sqrt() , (M - xs).sqrt())) 
    
    #coeff_tensors = []
    
    for j in range(n_points):
        poly_j = []
        for k in range(n_points): 
            if j==k:
                continue 
            else:
                y_jk = ps[j,k][1]
                if y_jk < 0:
                    poly_j.append(PolyM([-Xi[1][j,k], Xi[0][j,k]]))
                elif y_jk > 0:
                    poly_j.append(PolyM([ Xi[0][j,k], Xi[1][j,k]]))
                else:  # y_jk =0 
                    x_jk = ps[j,k][0]
                    if x_jk < 0:
                        poly_j.append(PolyM([Xi[0][j,k], Xi[1][j,k]]))
                    else:
                        poly_j.append(PolyM([Xi[0][j,k], Xi[1][j,k]]))
        
        prod_poly_j = reduce((lambda x, y: x * y), poly_j).values()
        coeffs = stack(prod_poly_j)        
        dots[j] = dot(coeffs, v) 
        
        #coeff_tensors.append(coeffs)
    
    #sample = torch.stack(coeff_tensors)
    sample = (p, v)

    classification = one_hot(dots) 
    return sample , classification 
