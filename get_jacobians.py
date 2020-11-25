#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:19:20 2020

@author: cornell
"""
import sympy 


sympy.init_printing(use_latex='mathjax')

px, py = sympy.symbols('px, py') #where the "wall" is

'''
wall = 3.
const_vel = 1.
m = 1.
'''
#'''
x_pos, y_pos, y_vel = sympy.symbols('x_pos, y_pos, y_vel')
u, dt = sympy.symbols('u, dt')
wall, const_vel, m = sympy.symbols('wall, const_vel, m')
w1, w2, w3 = sympy.symbols('w1, w2, w3')
v = sympy.symbols('v')

f = sympy.Matrix([ [x_pos + dt*const_vel + dt*w1], \
                   [y_pos + dt*y_vel + dt*dt/m/2*u + dt*w2], \
                   [y_vel + dt/m*u + dt*w3] ])
#h = sympy.Matrix([wall - y_pos])
h = sympy.Matrix([sympy.sqrt((px-x_pos)**2 + (py-y_pos)**2) + v])

z = sympy.Matrix([x_pos, y_pos, y_vel])
w = sympy.Matrix([w1, w2, w3])
inputs = sympy.Matrix([u])
sensor = sympy.Matrix([v])

F_x = f.jacobian(z)
F_u = f.jacobian(inputs)
G_w = f.jacobian(w)
H_x = h.jacobian(z)
H_v = h.jacobian(sensor)
#'''

''' 
OR
'''
'''
v, delta = sympy.symbols('v, delta')
x, y, theta = sympy.symbols('x, y, \Theta')
wall, const_vel, m = sympy.symbols('wall, const_vel, m')
w1, w2, w3, L = sympy.symbols('w1, w2, w3, L')

f = sympy.Matrix([ [v*sympy.cos(theta)+w1], \
                   [v*sympy.sin(theta)+w2], \
                   [v/L*sympy.tan(delta)+w3] ])
h = sympy.Matrix([wall - y])

z = sympy.Matrix([x, y, theta])
w = sympy.Matrix([w1, w2, w3])
inputs = sympy.Matrix([v, delta])

F_x = f.jacobian(z)
F_u = f.jacobian(inputs)
G_w = f.jacobian(w)
H_x = h.jacobian(z)
'''

print('Linearized matrices\n===================')
print('F_x = ', end='')
print(F_x)
print('F_u = ', end='')
print(F_u)
print('G_w = ', end='')
print(G_w)
print('H_x = ', end='')
print(H_x)
print('H_v = ', end='')
print(H_v)
