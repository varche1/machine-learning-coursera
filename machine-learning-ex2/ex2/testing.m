clear ; close all; clc

data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);

[m, n] = size(X);

X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

lambda = 1;

% Compute and display initial cost and gradient
[cost_1, grad_1] = costFunction(initial_theta, X, y)
[cost_2, grad_2] = costFunctionReg(initial_theta, X, y, lambda)