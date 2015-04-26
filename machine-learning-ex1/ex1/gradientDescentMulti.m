function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % theta_temp = theta;

    % for i = 1:length(X(1,:))
    %     q = @(n) (X(n,:)*theta - y(n) .* X(n, i));
    %     theta_temp(i) = theta(i) - (alpha/m) * sum(arrayfun(q,[1:m]));
    % end

    % theta = theta_temp;

    h = X * theta;  % m * 1 matrix
    errors = h .- y;  % m * 1 matrix
    decrement = (alpha / m) * (errors' * X);  % 1 * n
    theta = theta - decrement';  % 1 * n


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
