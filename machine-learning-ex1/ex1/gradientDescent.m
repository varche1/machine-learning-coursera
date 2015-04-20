function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %
    theta_temp = zeros(2, 1)

    for i = 1:m
        theta_temp(1) = X(i,:)*theta - y(i)
    end
    theta_temp(1) = theta(1) - alpha/m * theta_temp(1);

    for i = 1:m
        theta_temp(2) = (X(i,:)*theta - y(i)) * X(i,2)
    end
    theta_temp(2) = theta(2) - alpha/m * theta_temp(2);


    theta = theta_temp;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
